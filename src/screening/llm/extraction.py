"""CandidateExtractor: parses screening transcripts into structured profile updates."""

import json
import unicodedata
from time import perf_counter

from anthropic import Anthropic
from anthropic.types import MessageParam, OutputConfigParam
from pydantic import BaseModel, ConfigDict, Field

from screening.config import EXTRACTION_MAX_TOKENS, MODEL
from screening.domain.models import CandidateProfile
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


class FieldUpdate(BaseModel):
    """One structured update extracted from the screening transcript."""

    model_config = ConfigDict(extra="forbid")

    field: str
    value: str | None = None
    raw: str | None = None
    status: str | None = None
    years: float | None = None
    platform: str | None = None


class ProfileExtraction(BaseModel):
    """Structured LLM output containing candidate profile updates only."""

    model_config = ConfigDict(extra="forbid")

    updates: list[FieldUpdate] = Field(default_factory=list)


UPDATE_FIELD_NAMES = (
    "conversation_language",
    "full_name",
    "drivers_license",
    "city_zone",
    "availability",
    "preferred_schedule",
    "prior_delivery_experience",
    "start_date",
)


EXTRACTION_OUTPUT_CONFIG: OutputConfigParam = {
    "format": {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["updates"],
            "properties": {
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["field"],
                        "properties": {
                            "field": {
                                "type": "string",
                                "enum": list(UPDATE_FIELD_NAMES),
                            },
                            "value": {"type": "string"},
                            "raw": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "Matched",
                                    "Needs clarification",
                                    "Unsupported",
                                ],
                            },
                            "years": {"type": "number"},
                            "platform": {"type": "string"},
                        },
                    },
                }
            },
        },
    }
}

_ANSWER_STATUSES = frozenset({"Matched", "Needs clarification"})
_CITY_ZONE_STATUSES = frozenset({"Matched", "Needs clarification", "Unsupported"})


def _profile_from_extraction(extraction: ProfileExtraction) -> CandidateProfile:
    """Convert compact structured field updates into a CandidateProfile."""
    data: dict[str, object] = {}
    for update in extraction.updates:
        field = update.field
        if field == "conversation_language":
            _set_value(data, "conversation_language", update.value)
        elif field == "full_name":
            _set_value(data, "full_name", update.value)
            _set_answer_status(data, "full_name_status", update.status)
        elif field == "drivers_license":
            _set_value(data, "raw_drivers_license", update.raw)
            _set_value(data, "drivers_license", update.value)
        elif field == "city_zone":
            _set_value(data, "raw_city_zone", update.raw)
            _set_value(data, "city_zone", update.value)
            _set_city_status(data, update.status)
        elif field == "availability":
            _set_value(data, "raw_availability", update.raw)
            availability = update.value or _normalize_availability(update.raw)
            _set_value(data, "availability", availability)
            if availability is not None:
                data["availability_status"] = "Matched"
            else:
                _set_answer_status(data, "availability_status", update.status)
        elif field == "preferred_schedule":
            _set_value(data, "raw_preferred_schedule", update.raw)
            schedule = update.value or _normalize_preferred_schedule(update.raw)
            _set_value(data, "preferred_schedule", schedule)
            if schedule is not None:
                data["preferred_schedule_status"] = "Matched"
            else:
                _set_answer_status(data, "preferred_schedule_status", update.status)
        elif field == "prior_delivery_experience":
            _set_value(data, "raw_prior_delivery_experience", update.raw)
            experience = {}
            if update.years is not None:
                experience["years"] = update.years
            if update.platform is not None:
                experience["platform"] = update.platform
            if experience:
                data["prior_delivery_experience"] = experience
            _set_answer_status(
                data,
                "prior_delivery_experience_status",
                update.status,
            )
        elif field == "start_date":
            _set_value(data, "raw_start_date", update.raw)
            _set_value(data, "start_date", update.value)
            _set_answer_status(data, "start_date_status", update.status)
    return CandidateProfile.model_validate(data)


def _set_value(data: dict[str, object], key: str, value: str | None) -> None:
    if value is not None:
        data[key] = value


def _set_answer_status(data: dict[str, object], key: str, status: str | None) -> None:
    if status in _ANSWER_STATUSES:
        data[key] = status


def _set_city_status(data: dict[str, object], status: str | None) -> None:
    if status in _CITY_ZONE_STATUSES:
        data["city_zone_status"] = status


def _normalize_availability(value: str | None) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    if any(
        synonym in text
        for synonym in (
            "full time",
            "tiempo completo",
            "jornada completa",
            "full jornada",
        )
    ):
        return "Full-time"
    if any(
        synonym in text
        for synonym in (
            "part time",
            "medio tiempo",
            "media jornada",
            "parcial",
            "unas horas",
        )
    ):
        return "Part-time"
    if any(
        synonym in text
        for synonym in (
            "weekend",
            "finde",
            "findes",
            "fds",
            "fin de semana",
            "fines de semana",
            "sabados y domingos",
        )
    ):
        return "Weekends"
    return None


def _normalize_preferred_schedule(value: str | None) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    if any(
        synonym in text
        for synonym in ("morning", "manana", "temprano", "por la manana")
    ):
        return "Morning"
    if any(synonym in text for synonym in ("afternoon", "tarde", "por la tarde")):
        return "Afternoon"
    if any(
        synonym in text
        for synonym in ("evening", "night", "noche", "tarde noche", "por la noche")
    ):
        return "Evening"
    if any(
        synonym in text
        for synonym in ("flexible", "flex", "me adapto", "cualquiera", "indistinto")
    ):
        return "Flexible"
    return None


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    return " ".join(text.replace("-", " ").split()) or None


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
            updates_profile = _profile_from_extraction(updates)
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
