"""CandidateExtractor: parses screening transcripts into structured profile updates."""

import json

from anthropic import Anthropic
from anthropic.types import MessageParam

from screening.config import EXTRACTION_MAX_TOKENS, MODEL
from screening.domain.models import CandidateProfile
from screening.domain.service_areas import load_service_area_names
from screening.llm.prompts.extraction import EXTRACTION_SYSTEM_PROMPT
from screening.llm.utils import extract_text, parse_json_object


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
            json.JSONDecodeError: If the model response is not valid JSON.
        """
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=EXTRACTION_MAX_TOKENS,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": self._build_prompt(messages, current_profile),
                }
            ],
        )
        payload = parse_json_object(extract_text(response))
        updates = CandidateProfile.model_validate(payload)
        return current_profile.merge(updates)

    def _build_prompt(
        self,
        messages: list[MessageParam],
        current_profile: CandidateProfile,
    ) -> str:
        """Build the JSON user prompt for the extraction call.

        Bundles the supported service areas, the current profile, the
        transcript, and the required output shape into a JSON string.

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
                "required_output_shape": {
                    "conversation_language": None,
                    "full_name": None,
                    "raw_drivers_license": None,
                    "drivers_license": None,
                    "raw_city_zone": None,
                    "city_zone": None,
                    "city_zone_status": None,
                    "availability": None,
                    "preferred_schedule": None,
                    "prior_delivery_experience": {
                        "years": None,
                        "platform": None,
                    },
                    "start_date": None,
                },
            },
            ensure_ascii=True,
        )
