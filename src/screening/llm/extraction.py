"""CandidateExtractor: parses screening transcripts into structured profile updates."""

import json
from time import perf_counter
from typing import Any

from anthropic import Anthropic
from anthropic.types import MessageParam, OutputConfigParam
from pydantic import BaseModel, ConfigDict

from screening.config import EXTRACTION_MAX_TOKENS, MODEL
from screening.domain.models import (
    Availability,
    CandidateProfile,
    CityZoneStatus,
    ConversationLanguage,
    DeliveryExperience,
    DriverLicense,
    PreferredSchedule,
)
from screening.domain.service_areas import load_service_area_names
from screening.llm.prompts.extraction import EXTRACTION_SYSTEM_PROMPT
from screening.llm.utils import extract_text
from screening.observability import (
    anthropic_response_metadata,
    bind_context,
    exception_metadata,
    message_list_metadata,
    profile_state_metadata,
)


class ProfileExtraction(BaseModel):
    """Structured LLM output containing candidate profile updates only."""

    model_config = ConfigDict(extra="forbid")

    conversation_language: ConversationLanguage | None = None
    full_name: str | None = None
    raw_drivers_license: str | None = None
    drivers_license: DriverLicense | None = None
    raw_city_zone: str | None = None
    city_zone: str | None = None
    city_zone_status: CityZoneStatus | None = None
    availability: Availability | None = None
    preferred_schedule: PreferredSchedule | None = None
    prior_delivery_experience: DeliveryExperience | None = None
    start_date: str | None = None


def _build_extraction_schema() -> dict[str, Any]:
    """Return the extraction JSON schema with a dynamic service-area enum."""
    schema = ProfileExtraction.model_json_schema()
    city_schema = schema["properties"]["city_zone"]
    service_areas = list(load_service_area_names())

    for branch in city_schema.get("anyOf", []):
        if isinstance(branch, dict) and branch.get("type") == "string":
            branch["enum"] = service_areas
            break
    else:
        if city_schema.get("type") != "string":
            raise RuntimeError("city_zone schema does not contain a string branch")
        city_schema["enum"] = service_areas

    return schema


EXTRACTION_OUTPUT_CONFIG: OutputConfigParam = {
    "format": {
        "type": "json_schema",
        "schema": _build_extraction_schema(),
    }
}


class CandidateExtractor:
    """Extracts structured candidate profile updates from a transcript.

    Attributes:
        client (Anthropic): Anthropic API client used for the extraction call.
    """

    def __init__(self, client: Anthropic) -> None:
        """Initialize the extractor.

        Args:
            client (Anthropic): Anthropic API client used for extraction.
        """
        self.client = client

    def extract(
        self,
        messages: list[MessageParam],
        current_profile: CandidateProfile,
    ) -> CandidateProfile:
        """Extract profile updates from the transcript and merge them.

        Sends the transcript and current profile to the model, parses the JSON
        response into profile updates, and merges them onto the current
        profile.

        Args:
            messages (list[MessageParam]): The conversation transcript.
            current_profile (CandidateProfile): The profile to update.

        Returns:
            CandidateProfile: The merged, re-validated profile.

        Raises:
            ValueError: If the model response does not match the schema.
        """
        started_at = perf_counter()
        prompt = self._build_prompt(messages, current_profile)
        bind_context(
            event="llm_call_started",
            operation="extraction",
            model=MODEL,
            max_output_tokens=EXTRACTION_MAX_TOKENS,
            prompt_char_count=len(prompt),
            **message_list_metadata(list(messages)),
        ).info("Extraction LLM call started")

        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=EXTRACTION_MAX_TOKENS,
                system=EXTRACTION_SYSTEM_PROMPT,
                output_config=EXTRACTION_OUTPUT_CONFIG,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            text = extract_text(response)
            updates = ProfileExtraction.model_validate_json(text)
            updates_profile = CandidateProfile.model_validate(updates.model_dump())
            merged_profile = current_profile.merge(updates_profile)
        except Exception as error:
            bind_context(
                event="llm_call_failed",
                operation="extraction",
                model=MODEL,
                duration_ms=_duration_ms(started_at),
                **exception_metadata(error),
            ).exception("Extraction LLM call failed")
            raise

        bind_context(
            event="llm_call_completed",
            operation="extraction",
            model=MODEL,
            duration_ms=_duration_ms(started_at),
            output_char_count=len(text),
            **anthropic_response_metadata(response),
            **profile_state_metadata(merged_profile),
        ).info("Extraction LLM call completed")
        return merged_profile

    def _build_prompt(
        self,
        messages: list[MessageParam],
        current_profile: CandidateProfile,
    ) -> str:
        """Build the JSON user prompt for the extraction call.

        Bundles the supported service areas, the current profile, and the
        transcript into a JSON string.

        Args:
            messages (list[MessageParam]): The conversation transcript.
            current_profile (CandidateProfile): The current profile state.

        Returns:
            str: A JSON-encoded prompt payload.
        """
        return json.dumps(
            {
                "service_areas": load_service_area_names(),
                "current_profile": current_profile.model_dump(mode="json"),
                "transcript": messages,
            },
            ensure_ascii=True,
        )


def _duration_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)
