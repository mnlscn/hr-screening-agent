import json

from anthropic import Anthropic
from anthropic.types import MessageParam

from screening.config import EXTRACTION_MAX_TOKENS, MODEL
from screening.models import CandidateProfile
from screening.prompt import EXTRACTION_SYSTEM_PROMPT
from screening.utils import extract_text, load_service_area_names


class CandidateExtractor:
    def __init__(self, client: Anthropic) -> None:
        self.client = client

    def extract(
        self,
        messages: list[MessageParam],
        current_profile: CandidateProfile,
    ) -> CandidateProfile:
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
        return json.dumps(
            {
                "service_areas": load_service_area_names(),
                "current_profile": current_profile.model_dump(mode="json"),
                "transcript": messages,
                "required_output_shape": {
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


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)
