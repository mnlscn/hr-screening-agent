"""Tests for CandidateExtractor: structured output schema and profile updates."""

import json
from dataclasses import dataclass
from typing import Any, cast

from anthropic.types import MessageParam

from screening.domain.models import CandidateProfile, DeliveryExperience
from screening.domain.service_areas import load_service_area_names
from screening.llm.extraction import CandidateExtractor, EXTRACTION_OUTPUT_CONFIG


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


def extraction_output_format() -> dict[str, Any]:
    output_format = EXTRACTION_OUTPUT_CONFIG.get("format")
    assert output_format is not None
    return cast(dict[str, Any], output_format)


def extraction_schema() -> dict[str, Any]:
    return cast(dict[str, Any], extraction_output_format()["schema"])


def city_zone_string_schema() -> dict[str, Any]:
    city_zone = extraction_schema()["properties"]["city_zone"]
    for branch in city_zone["anyOf"]:
        if branch.get("type") == "string":
            return branch
    raise AssertionError("city_zone schema has no string branch")


def test_extractor_prompt_includes_context_without_output_shape():
    extractor = CandidateExtractor(client=cast(Any, object()))

    payload = json.loads(
        extractor._build_prompt(
            messages=[{"role": "user", "content": "I would prefer English"}],
            current_profile=CandidateProfile(),
        )
    )

    assert payload["service_areas"] == list(load_service_area_names())
    assert payload["current_profile"]["conversation_language"] is None
    assert payload["transcript"] == [
        {"role": "user", "content": "I would prefer English"}
    ]
    assert "required_output_shape" not in payload


def test_extraction_output_config_uses_strict_json_schema():
    assert extraction_output_format()["type"] == "json_schema"
    schema = extraction_schema()

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["DeliveryExperience"]["additionalProperties"] is False
    assert city_zone_string_schema()["enum"] == list(load_service_area_names())
    assert len(city_zone_string_schema()["enum"]) == 45


def test_extract_uses_structured_output_and_merges_profile():
    client = FakeClient(
        json.dumps(
            {
                "conversation_language": "Spanish",
                "full_name": "Maria Garcia",
                "raw_drivers_license": "si",
                "drivers_license": "Yes",
                "raw_city_zone": "Madrid",
                "city_zone": "Madrid",
                "city_zone_status": "Matched",
                "availability": "Full-time",
                "preferred_schedule": "Morning",
                "prior_delivery_experience": {
                    "years": 2,
                    "platform": "Glovo",
                },
                "start_date": "next Monday",
            }
        )
    )
    extractor = CandidateExtractor(client=cast(Any, client))

    profile = extractor.extract(
        messages=[cast(MessageParam, {"role": "user", "content": "Soy Maria Garcia"})],
        current_profile=CandidateProfile(),
    )

    call = client.messages.calls[0]
    assert call["output_config"] == EXTRACTION_OUTPUT_CONFIG
    assert profile.full_name == "Maria Garcia"
    assert profile.drivers_license == "Yes"
    assert profile.city_zone == "Madrid"
    assert profile.prior_delivery_experience == DeliveryExperience(
        years=2,
        platform="Glovo",
    )


def test_extract_null_experience_does_not_overwrite_existing_experience():
    client = FakeClient(json.dumps({"prior_delivery_experience": None}))
    extractor = CandidateExtractor(client=cast(Any, client))
    current_profile = CandidateProfile(
        prior_delivery_experience=DeliveryExperience(years=1, platform="Uber Eats")
    )

    profile = extractor.extract(
        messages=[cast(MessageParam, {"role": "user", "content": "Ok"})],
        current_profile=current_profile,
    )

    assert profile.prior_delivery_experience == DeliveryExperience(
        years=1,
        platform="Uber Eats",
    )
