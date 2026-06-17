import json

from anthropic import Anthropic

from screening.config import EXTRACTION_MAX_TOKENS, MODEL, SERVICE_AREAS
from screening.models import CandidateProfile
from screening.utils import extract_text


EXTRACTION_SYSTEM_PROMPT = """\
You extract structured recruiting screening data from a delivery-driver chat.
Return strict JSON only. Do not include markdown, comments, or explanations.
Use null when a field is unknown.

Allowed values:
- drivers_license: "Yes", "No", or null
- availability: "Full-time", "Part-time", "Weekends", or null
- preferred_schedule: "Morning", "Afternoon", "Evening", "Flexible", or null

prior_delivery_experience must be an object with:
- years: number or null
- platform: string or null
"""


class CandidateExtractor:
    def __init__(self, client: Anthropic) -> None:
        self.client = client

    def extract(
        self,
        messages: list[dict[str, str]],
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
        messages: list[dict[str, str]],
        current_profile: CandidateProfile,
    ) -> str:
        return json.dumps(
            {
                "service_areas": SERVICE_AREAS,
                "current_profile": current_profile.model_dump(mode="json"),
                "transcript": messages,
                "required_output_shape": {
                    "full_name": None,
                    "drivers_license": None,
                    "city_zone": None,
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

