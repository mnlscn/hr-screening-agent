"""Tests for CandidateExtractor: structured output schema and profile updates."""

import json
from dataclasses import dataclass
from typing import Any, cast

import pytest
from anthropic.types import MessageParam

from screening.domain.models import CandidateProfile, DeliveryExperience
from screening.domain.service_areas import load_service_area_names
from screening.llm.extraction import (
    EXTRACTION_OUTPUT_CONFIG,
    CandidateExtractor,
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


def extraction_output_format() -> dict[str, Any]:
    output_format = EXTRACTION_OUTPUT_CONFIG.get("format")
    assert output_format is not None
    return cast(dict[str, Any], output_format)


def extraction_schema() -> dict[str, Any]:
    return cast(dict[str, Any], extraction_output_format()["schema"])


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
    schema_text = json.dumps(schema)
    properties = schema["properties"]

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["updates"]
    assert "$defs" not in schema
    assert "$ref" not in schema_text
    assert "default" not in schema_text
    assert "title" not in schema_text
    assert "description" not in schema_text
    assert "anyOf" not in schema_text
    assert schema_text.count('"type"') < 12
    updates = properties["updates"]
    assert updates["type"] == "array"
    item = updates["items"]
    assert item["additionalProperties"] is False
    assert item["required"] == ["field"]
    assert item["properties"]["field"]["enum"] == [
        "conversation_language",
        "full_name",
        "drivers_license",
        "city_zone",
        "availability",
        "preferred_schedule",
        "prior_delivery_experience",
        "start_date",
    ]
    assert item["properties"]["status"]["enum"] == [
        "Matched",
        "Needs clarification",
        "Unsupported",
    ]


def test_extract_uses_structured_output_and_merges_profile():
    client = FakeClient(
        json.dumps(
            {
                "updates": [
                    {"field": "conversation_language", "value": "Spanish"},
                    {
                        "field": "full_name",
                        "value": "Maria Garcia",
                        "status": "Matched",
                    },
                    {"field": "drivers_license", "raw": "si", "value": "Yes"},
                    {
                        "field": "city_zone",
                        "raw": "Madrid",
                        "value": "Madrid",
                        "status": "Matched",
                    },
                    {
                        "field": "availability",
                        "raw": "tiempo completo",
                        "value": "Full-time",
                        "status": "Matched",
                    },
                    {
                        "field": "preferred_schedule",
                        "raw": "manana",
                        "value": "Morning",
                        "status": "Matched",
                    },
                    {
                        "field": "prior_delivery_experience",
                        "raw": "2 anos en Glovo",
                        "years": 2,
                        "platform": "Glovo",
                        "status": "Matched",
                    },
                    {
                        "field": "start_date",
                        "raw": "next Monday",
                        "value": "next Monday",
                        "status": "Matched",
                    },
                ]
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
    assert profile.raw_availability == "tiempo completo"
    assert profile.availability_status == "Matched"
    assert profile.prior_delivery_experience == DeliveryExperience(
        years=2,
        platform="Glovo",
    )


def test_extract_omitted_experience_does_not_overwrite_existing_experience():
    client = FakeClient(json.dumps({"updates": []}))
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


def test_extract_normalizes_high_confidence_raw_availability_alias():
    client = FakeClient(
        json.dumps(
            {
                "updates": [
                    {
                        "field": "availability",
                        "raw": "findes",
                        "status": "Needs clarification",
                    }
                ]
            }
        )
    )
    extractor = CandidateExtractor(client=cast(Any, client))

    profile = extractor.extract(
        messages=[cast(MessageParam, {"role": "user", "content": "findes"})],
        current_profile=CandidateProfile(),
    )

    assert profile.availability == "Weekends"
    assert profile.availability_status == "Matched"
    assert "availability" not in profile.clarification_fields


def test_extract_rejects_non_canonical_city_zone_locally():
    client = FakeClient(
        json.dumps(
            {
                "updates": [
                    {
                        "field": "city_zone",
                        "raw": "Paris",
                        "value": "Paris",
                        "status": "Matched",
                    }
                ]
            }
        )
    )
    extractor = CandidateExtractor(client=cast(Any, client))

    with pytest.raises(ValueError, match="city_zone must be a canonical service area"):
        extractor.extract(
            messages=[cast(MessageParam, {"role": "user", "content": "Paris"})],
            current_profile=CandidateProfile(),
        )
