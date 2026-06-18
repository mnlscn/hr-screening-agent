import json
from dataclasses import dataclass
from typing import Any, cast

import pytest
from anthropic.types import MessageParam

from screening.models import CandidateProfile, DeliveryExperience
from screening.summary import (
    CandidateSummarizer,
    determine_bot_label,
)


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeResponse:
    content: list[TextBlock]


class FakeMessages:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(content=[TextBlock(self.response_text)])


class FakeClient:
    def __init__(self, response_text: str):
        self.messages = FakeMessages(response_text)


MARKDOWN_HR_SUMMARY = """\
**Candidate Brief**
Maria Garcia is applying for delivery work in Madrid.

**Screening Facts**
- License: Valid driver license reported.
- Availability: Full-time, morning schedule.
- Experience: 2 years with Glovo.

**Conversation Notes**
- Candidate replied politely and clearly.

**Follow-Up Suggestions**
1. Confirm start date during onboarding.
"""


def complete_profile() -> CandidateProfile:
    return CandidateProfile(
        full_name="Maria Garcia",
        drivers_license="Yes",
        raw_city_zone="Madrid",
        city_zone="Madrid",
        city_zone_status="Matched",
        conversation_language="Spanish",
        availability="Full-time",
        preferred_schedule="Morning",
        prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
        start_date="next Monday",
    )


def test_determine_bot_label_for_complete_profile():
    assert determine_bot_label(complete_profile()) == "eligible"


def test_determine_bot_label_for_disqualified_profile():
    assert determine_bot_label(CandidateProfile(drivers_license="No")) == (
        "not_eligible"
    )


def test_determine_bot_label_for_incomplete_or_failed_extraction():
    assert determine_bot_label(CandidateProfile(full_name="Luis Perez")) == (
        "needs_review"
    )
    assert determine_bot_label(complete_profile(), extraction_failed=True) == (
        "needs_review"
    )


def test_summarizer_uses_metadata_and_full_transcript():
    client = FakeClient(
        json.dumps({"bot_label": "eligible", "hr_summary": MARKDOWN_HR_SUMMARY})
    )
    summarizer = CandidateSummarizer(client=cast(Any, client))
    transcript = [
        cast(MessageParam, {"role": "assistant", "content": "Hola"}),
        cast(MessageParam, {"role": "user", "content": "Gracias, soy Maria Garcia"}),
    ]

    summary = summarizer.summarize(complete_profile(), transcript)

    assert summary.bot_label == "eligible"
    assert summary.hr_summary.startswith("**Candidate Brief**")
    assert "**Screening Facts**" in summary.hr_summary
    assert "**Conversation Notes**" in summary.hr_summary
    assert "**Follow-Up Suggestions**" in summary.hr_summary
    assert "replied politely" in summary.hr_summary
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-4-6"
    prompt = call["messages"][0]["content"]
    assert "extracted_metadata" in prompt
    assert "full_transcript" in prompt
    assert "Gracias, soy Maria Garcia" in prompt
    assert "politeness" in call["system"]
    assert "**Candidate Brief**" in call["system"]
    assert "**Screening Facts**" in call["system"]


def test_summarizer_keeps_incomplete_profile_as_needs_review():
    client = FakeClient(
        json.dumps(
            {
                "bot_label": "eligible",
                "hr_summary": (
                    "**Candidate Brief**\n"
                    "Luis Perez has started the screening but the profile is "
                    "incomplete.\n\n"
                    "**Screening Facts**\n"
                    "- Missing required screening information.\n\n"
                    "**Conversation Notes**\n"
                    "- Candidate replied clearly.\n\n"
                    "**Follow-Up Suggestions**\n"
                    "1. Collect the missing fields before HR review."
                ),
            }
        )
    )
    summarizer = CandidateSummarizer(client=cast(Any, client))

    summary = summarizer.summarize(
        CandidateProfile(full_name="Luis Perez"),
        [cast(MessageParam, {"role": "user", "content": "Soy Luis Perez"})],
    )

    assert summary.bot_label == "needs_review"


def test_summarizer_rejects_invalid_json():
    client = FakeClient("not json")
    summarizer = CandidateSummarizer(client=cast(Any, client))

    with pytest.raises(ValueError):
        summarizer.summarize(complete_profile(), [])
