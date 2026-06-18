"""CandidateSummarizer: generates HR summaries and bot labels from completed screenings."""

import json
from dataclasses import dataclass
from time import perf_counter

from anthropic import Anthropic
from anthropic.types import MessageParam, OutputConfigParam
from pydantic import BaseModel, ConfigDict, field_validator

from screening.config import SUMMARY_MAX_TOKENS, SUMMARY_MODEL
from screening.domain.models import BotLabel, CandidateProfile
from screening.llm.prompts.summary import SUMMARY_SYSTEM_PROMPT
from screening.llm.utils import extract_text
from screening.observability import (
    anthropic_response_metadata,
    bind_context,
    exception_metadata,
    message_list_metadata,
    profile_state_metadata,
)


class CandidateSummaryPayload(BaseModel):
    """Structured LLM output for a candidate summary.

    Attributes:
        bot_label (BotLabel): Model-proposed triage label.
        hr_summary (str): Markdown recruiter-facing summary text.
    """

    model_config = ConfigDict(extra="forbid")

    bot_label: BotLabel
    hr_summary: str

    @field_validator("hr_summary")
    @classmethod
    def validate_summary_text(cls, value: str) -> str:
        """Validate and normalize the generated summary text.

        Args:
            value (str): Raw summary text from the structured model output.

        Returns:
            str: Stripped, non-empty summary text.

        Raises:
            ValueError: If the summary is empty.
        """
        value = value.strip()
        if not value:
            raise ValueError("hr_summary must be a non-empty string")
        return value


SUMMARY_OUTPUT_CONFIG: OutputConfigParam = {
    "format": {
        "type": "json_schema",
        "schema": CandidateSummaryPayload.model_json_schema(),
    }
}


@dataclass(frozen=True)
class CandidateSummary:
    """Generated HR summary and triage label for a candidate.

    Attributes:
        bot_label (BotLabel): Triage label: "eligible", "not_eligible", or
            "needs_review".
        hr_summary (str): Markdown HR-facing summary text.
        model (str): Identifier of the model that produced the summary;
            defaults to ``SUMMARY_MODEL``.
    """

    bot_label: BotLabel
    hr_summary: str
    model: str = SUMMARY_MODEL


class CandidateSummarizer:
    """Generates HR summaries and triage labels from completed screenings.

    Attributes:
        client (Anthropic): Anthropic API client used for the summary call.
    """

    def __init__(self, client: Anthropic) -> None:
        """Initialize the summarizer.

        Args:
            client (Anthropic): Anthropic API client used for summarization.
        """
        self.client = client

    def summarize(
        self,
        profile: CandidateProfile,
        transcript: list[MessageParam],
        *,
        extraction_failed: bool = False,
    ) -> CandidateSummary:
        """Generate an HR summary and triage label for a candidate.

        Computes a deterministic baseline label, asks the model for a summary
        and label, validates the response, and reconciles the model label
        against the baseline.

        Args:
            profile (CandidateProfile): The extracted candidate profile.
            transcript (list[MessageParam]): The full conversation transcript.
            extraction_failed (bool): Whether profile extraction failed during
                the session.

        Returns:
            CandidateSummary: The reconciled label and HR summary text.

        Raises:
            ValueError: If the model response does not match the structured
                summary schema.
        """
        baseline_label = determine_bot_label(
            profile,
            extraction_failed=extraction_failed,
        )
        started_at = perf_counter()
        prompt = self._build_prompt(
            profile,
            transcript,
            baseline_label=baseline_label,
            extraction_failed=extraction_failed,
        )
        bind_context(
            event="llm_call_started",
            operation="summary",
            model=SUMMARY_MODEL,
            max_output_tokens=SUMMARY_MAX_TOKENS,
            baseline_bot_label=baseline_label,
            extraction_failed=extraction_failed,
            prompt_char_count=len(prompt),
            **message_list_metadata(list(transcript)),
            **profile_state_metadata(profile),
        ).info("Summary LLM call started")

        try:
            response = self.client.messages.create(
                model=SUMMARY_MODEL,
                max_tokens=SUMMARY_MAX_TOKENS,
                system=SUMMARY_SYSTEM_PROMPT,
                output_config=SUMMARY_OUTPUT_CONFIG,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            text = extract_text(response)
            payload = CandidateSummaryPayload.model_validate_json(text)
        except Exception as error:
            bind_context(
                event="llm_call_failed",
                operation="summary",
                model=SUMMARY_MODEL,
                duration_ms=_duration_ms(started_at),
                **exception_metadata(error),
            ).exception("Summary LLM call failed")
            raise

        summary = CandidateSummary(
            bot_label=_resolve_bot_label(profile, baseline_label, payload.bot_label),
            hr_summary=payload.hr_summary,
        )
        bind_context(
            event="llm_call_completed",
            operation="summary",
            model=SUMMARY_MODEL,
            duration_ms=_duration_ms(started_at),
            output_char_count=len(text),
            hr_summary_char_count=len(payload.hr_summary),
            model_bot_label=payload.bot_label,
            final_bot_label=summary.bot_label,
            **anthropic_response_metadata(response),
        ).info("Summary LLM call completed")
        return summary

    def _build_prompt(
        self,
        profile: CandidateProfile,
        transcript: list[MessageParam],
        *,
        baseline_label: BotLabel,
        extraction_failed: bool,
    ) -> str:
        """Build the JSON user prompt for the summary call.

        Bundles the extracted metadata, extraction state, full transcript,
        labeling rules, and required output shape into a JSON string.

        Args:
            profile (CandidateProfile): The extracted candidate profile.
            transcript (list[MessageParam]): The full conversation transcript.
            baseline_label (BotLabel): The deterministic baseline triage label.
            extraction_failed (bool): Whether profile extraction failed.

        Returns:
            str: A JSON-encoded prompt payload.
        """
        return json.dumps(
            {
                "extracted_metadata": profile.model_dump(mode="json"),
                "extraction_state": {
                    "extraction_failed": extraction_failed,
                    "is_complete": profile.is_complete,
                    "is_disqualified": profile.is_disqualified,
                    "missing_fields": profile.missing_fields,
                    "clarification_fields": profile.clarification_fields,
                    "disqualification_reasons": profile.disqualification_reasons,
                    "baseline_bot_label": baseline_label,
                },
                "full_transcript": transcript,
                "bot_label_rules": {
                    "eligible": (
                        "Metadata is complete, no clarification fields, and no "
                        "hard rule failures."
                    ),
                    "not_eligible": (
                        "Metadata shows a hard rule failure, such as no valid "
                        "driver license or unsupported work location."
                    ),
                    "needs_review": (
                        "Extraction failed, metadata is incomplete, fields need "
                        "clarification, transcript contradicts metadata, or you "
                        "are unsure."
                    ),
                },
                "required_output_shape": {
                    "bot_label": baseline_label,
                    "hr_summary": None,
                },
            },
            ensure_ascii=True,
        )


def determine_bot_label(
    profile: CandidateProfile,
    *,
    extraction_failed: bool = False,
) -> BotLabel:
    """Compute a deterministic triage label from profile state.

    Args:
        profile (CandidateProfile): The extracted candidate profile.
        extraction_failed (bool): Whether profile extraction failed.

    Returns:
        BotLabel: "needs_review" when extraction failed or the profile is
            incomplete, "not_eligible" on a disqualification, or "eligible"
            when complete with no clarifications.
    """
    if extraction_failed:
        return "needs_review"
    if profile.disqualification_reasons:
        return "not_eligible"
    if profile.is_complete and not profile.clarification_fields:
        return "eligible"
    return "needs_review"


def _resolve_bot_label(
    profile: CandidateProfile,
    baseline_label: BotLabel,
    model_label: BotLabel,
) -> BotLabel:
    """Reconcile the model's label against the deterministic baseline.

    Guards against the model overriding a "needs_review" or "not_eligible"
    baseline that the profile state still supports.

    Args:
        profile (CandidateProfile): The extracted candidate profile.
        baseline_label (BotLabel): The deterministic baseline label.
        model_label (BotLabel): The label proposed by the model.

    Returns:
        BotLabel: The final reconciled triage label.
    """
    if baseline_label == "needs_review" and not profile.disqualification_reasons:
        return "needs_review"
    if baseline_label == "not_eligible" and model_label == "eligible":
        return "needs_review"
    return model_label


def _duration_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)
