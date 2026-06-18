import json
from dataclasses import dataclass
from typing import Literal, cast

from anthropic import Anthropic
from anthropic.types import MessageParam

from screening.config import SUMMARY_MAX_TOKENS, SUMMARY_MODEL
from screening.extraction import parse_json_object
from screening.models import CandidateProfile
from screening.utils import extract_text


BotLabel = Literal["eligible", "not_eligible", "needs_review"]
VALID_BOT_LABELS = {"eligible", "not_eligible", "needs_review"}

SUMMARY_SYSTEM_PROMPT = """\
You create concise HR screening summaries for delivery-driver candidates.
Return strict JSON only. Do not include markdown, comments, or explanations.

The human HR operator makes the hiring or discard decision. Your job is decision
support only: summarize evidence, nuance, and follow-up points from the transcript
and extracted metadata.

Use only evidence from the provided transcript and metadata. Do not infer protected
attributes, personality traits beyond observable conversation behavior, health,
family status, nationality, immigration status, age, religion, or other sensitive
personal information.

Output JSON shape:
{
  "bot_label": "eligible | not_eligible | needs_review",
  "hr_summary": "brief HR-facing summary"
}

For hr_summary, include the key screening facts, eligibility context, useful
conversation nuance such as politeness, clarity, responsiveness, motivation,
confusion, contradictions, hesitation, and concrete follow-up suggestions for HR.
Keep it brief, specific, and evidence-based.
""".strip()


@dataclass(frozen=True)
class CandidateSummary:
    bot_label: BotLabel
    hr_summary: str
    model: str = SUMMARY_MODEL


class CandidateSummarizer:
    def __init__(self, client: Anthropic) -> None:
        self.client = client

    def summarize(
        self,
        profile: CandidateProfile,
        transcript: list[MessageParam],
        *,
        extraction_failed: bool = False,
    ) -> CandidateSummary:
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
    if extraction_failed:
        return "needs_review"
    if profile.disqualification_reasons:
        return "not_eligible"
    if profile.is_complete and not profile.clarification_fields:
        return "eligible"
    return "needs_review"


def _validate_bot_label(value: object) -> BotLabel:
    if not isinstance(value, str) or value not in VALID_BOT_LABELS:
        raise ValueError(f"invalid bot_label: {value!r}")
    return cast(BotLabel, value)


def _validate_summary_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("hr_summary must be a non-empty string")
    return value.strip()


def _resolve_bot_label(
    profile: CandidateProfile,
    baseline_label: BotLabel,
    model_label: BotLabel,
) -> BotLabel:
    if baseline_label == "needs_review" and not profile.disqualification_reasons:
        return "needs_review"
    if baseline_label == "not_eligible" and model_label == "eligible":
        return "needs_review"
    return model_label
