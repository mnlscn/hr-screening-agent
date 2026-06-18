"""CandidateSummarizer: generates HR summaries and bot labels from completed screenings."""

import json
from dataclasses import dataclass
from typing import cast

from anthropic import Anthropic
from anthropic.types import MessageParam

from screening.config import SUMMARY_MAX_TOKENS, SUMMARY_MODEL
from screening.domain.models import VALID_BOT_LABELS, BotLabel, CandidateProfile
from screening.llm.prompts.summary import SUMMARY_SYSTEM_PROMPT
from screening.llm.utils import extract_text, parse_json_object


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
            ValueError: If the model response has an invalid label or an empty
                summary.
            json.JSONDecodeError: If the model response is not valid JSON.
        """
        baseline_label = determine_bot_label(
            profile,
            extraction_failed=extraction_failed,
        )
        response = self.client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=SUMMARY_MAX_TOKENS,
            system=SUMMARY_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": self._build_prompt(
                        profile,
                        transcript,
                        baseline_label=baseline_label,
                        extraction_failed=extraction_failed,
                    ),
                }
            ],
        )
        payload = parse_json_object(extract_text(response))
        bot_label = _validate_bot_label(payload.get("bot_label"))
        hr_summary = _validate_summary_text(payload.get("hr_summary"))

        return CandidateSummary(
            bot_label=_resolve_bot_label(profile, baseline_label, bot_label),
            hr_summary=hr_summary,
        )

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


def _validate_bot_label(value: object) -> BotLabel:
    """Validate that a value is a recognized triage label.

    Args:
        value (object): Candidate label value from the model response.

    Returns:
        BotLabel: The validated label.

    Raises:
        ValueError: If the value is not one of the valid bot labels.
    """
    if not isinstance(value, str) or value not in VALID_BOT_LABELS:
        raise ValueError(f"invalid bot_label: {value!r}")
    return cast(BotLabel, value)


def _validate_summary_text(value: object) -> str:
    """Validate that a value is a non-empty summary string.

    Args:
        value (object): Candidate summary text from the model response.

    Returns:
        str: The stripped, non-empty summary text.

    Raises:
        ValueError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("hr_summary must be a non-empty string")
    return value.strip()


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
