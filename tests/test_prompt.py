from screening.models import CandidateProfile, DeliveryExperience
from screening.prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    OPENING_MESSAGE,
    build_system_prompt,
    get_next_field,
)


def test_opening_message_matches_recruiter_outreach_flow():
    assert "solicitud" in OPENING_MESSAGE
    assert "conductor/a de reparto" in OPENING_MESSAGE
    assert "asistente de IA" in OPENING_MESSAGE
    assert OPENING_MESSAGE.count("?") == 1


def test_extraction_prompt_lives_with_prompts():
    assert "Return strict JSON only" in EXTRACTION_SYSTEM_PROMPT
    assert "service_areas" in EXTRACTION_SYSTEM_PROMPT
    assert "drivers_license" in EXTRACTION_SYSTEM_PROMPT


def test_missing_fields_drive_next_question_goal():
    profile = CandidateProfile(
        full_name="Giacomo Ortiz",
        drivers_license="Yes",
        raw_city_zone="Guadalajara",
        city_zone="Guadalajara Centro",
        city_zone_status="Matched",
        prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
        start_date="next week",
    )

    assert get_next_field(profile) == "availability"

    prompt = build_system_prompt(profile)

    assert '"next_field_to_collect": "availability"' in prompt
    assert "Do not ask for phone number, email" in prompt
    assert "Do not close or wrap up" in prompt


def test_clarification_fields_are_prioritized():
    profile = CandidateProfile(
        full_name="Giacomo Ortiz",
        raw_drivers_license="la mexicana la tendre proxima semana",
        drivers_license="Pending",
        raw_city_zone="Guadalajara",
        city_zone="Guadalajara Centro",
        city_zone_status="Matched",
    )

    assert get_next_field(profile) == "drivers_license"

    prompt = build_system_prompt(profile)

    assert '"next_field_to_collect": "drivers_license"' in prompt
    assert "valid local driver's license before starting" in prompt
