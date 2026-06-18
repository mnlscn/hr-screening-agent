"""Tests for the agent system prompt builder and language-detection helpers."""

from screening.domain.models import CandidateProfile, DeliveryExperience
from screening.llm.prompts.agent import (
    OPENING_MESSAGE,
    SYSTEM_PROMPT,
    build_system_prompt,
    detect_latest_user_language,
    get_next_field,
)
from screening.llm.prompts.extraction import EXTRACTION_SYSTEM_PROMPT


def test_opening_message_matches_recruiter_outreach_flow():
    assert "solicitud" in OPENING_MESSAGE
    assert "hacer repartos" in OPENING_MESSAGE
    assert "conductor/a" not in OPENING_MESSAGE
    assert "México o España" in OPENING_MESSAGE
    assert "hablemos en español" in OPENING_MESSAGE
    assert "asistente de IA" in OPENING_MESSAGE
    assert OPENING_MESSAGE.count("?") == 1


def test_extraction_prompt_lives_with_prompts():
    assert "Return strict JSON only" in EXTRACTION_SYSTEM_PROMPT
    assert "service_areas" in EXTRACTION_SYSTEM_PROMPT
    assert "drivers_license" in EXTRACTION_SYSTEM_PROMPT
    assert "conversation_language" in EXTRACTION_SYSTEM_PROMPT
    assert '"English", "Spanish", "Mixed"' in EXTRACTION_SYSTEM_PROMPT
    assert "valid for driving in Spain or Mexico" in EXTRACTION_SYSTEM_PROMPT
    assert "Any driving license issued by an EU country is valid" in (
        EXTRACTION_SYSTEM_PROMPT
    )
    assert "Do not require one vehicle type over another" in EXTRACTION_SYSTEM_PROMPT
    assert "set years to 0 and platform to null" in EXTRACTION_SYSTEM_PROMPT


def test_chat_prompt_handles_code_switching():
    assert "code-switch" in SYSTEM_PROMPT
    assert "dominant language of their latest message" in SYSTEM_PROMPT
    assert "answer in Spanish" in SYSTEM_PROMPT


def test_prompt_instructs_english_reply_for_english_latest_message():
    prompt = build_system_prompt(
        CandidateProfile(full_name="Marco"),
        latest_user_message="my name is marco",
    )

    assert '"latest_user_language": "English"' in prompt
    assert "Reply in English for the next assistant message" in prompt


def test_latest_user_language_detects_code_switching():
    assert detect_latest_user_language("Sí, I can start next week") == "Mixed"


def test_missing_fields_drive_next_question_goal():
    profile = CandidateProfile(
        full_name="Giacomo Ortiz",
        drivers_license="Yes",
        raw_city_zone="Guadalajara",
        city_zone="Guadalajara",
        city_zone_status="Matched",
        prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
        start_date="next week",
    )

    assert get_next_field(profile) == "availability"

    prompt = build_system_prompt(profile)

    assert '"next_field_to_collect": "availability"' in prompt
    assert "Do not ask for phone number, email" in prompt
    assert "Do not close or wrap up" in prompt
    assert "Service areas are internal" in prompt


def test_single_name_drives_surname_follow_up():
    profile = CandidateProfile(full_name="Marco")

    assert get_next_field(profile) == "full_name"

    prompt = build_system_prompt(profile, latest_user_message="my name is marco")

    assert '"next_field_to_collect": "full_name"' in prompt
    assert "ask for their surname or last name" in prompt


def test_service_area_questions_are_deflected_in_chat_prompt():
    profile = CandidateProfile(
        full_name="Giacomo Ortiz",
        drivers_license="Yes",
    )

    prompt = build_system_prompt(profile)

    assert "Never list, suggest, confirm, or deny available service areas" in prompt
    assert "ask which city or zone in Spain or Mexico they want to work in" in prompt
    assert (
        "Do not say it works, is supported, is available, or is not available" in prompt
    )


def test_license_and_city_questions_are_scoped_to_spain_and_mexico():
    prompt = build_system_prompt(CandidateProfile(full_name="Giacomo Ortiz"))

    assert "ask only whether they have a driver's license valid" in prompt
    assert "city or zone in Spain or Mexico" in prompt


def test_eu_and_vehicle_license_types_are_accepted_in_chat_prompt():
    prompt = build_system_prompt(CandidateProfile(full_name="Giacomo Ortiz"))

    assert "Any driving license issued by an EU country is valid" in prompt
    assert "Car, truck, and motorbike licenses are all acceptable" in prompt
    assert "Do not ask which of those vehicle types their license is for" in prompt
    assert "never say motorbike is not accepted" in prompt


def test_no_delivery_experience_does_not_trigger_extra_driving_questions():
    prompt = build_system_prompt(CandidateProfile(full_name="Giacomo Ortiz"))

    assert "accept that answer and move to the next needed field" in prompt
    assert "Do not ask for other driving or vehicle experience" in prompt


def test_clarification_fields_are_prioritized():
    profile = CandidateProfile(
        full_name="Giacomo Ortiz",
        raw_drivers_license="la mexicana la tendre proxima semana",
        drivers_license="Pending",
        raw_city_zone="Guadalajara",
        city_zone="Guadalajara",
        city_zone_status="Matched",
    )

    assert get_next_field(profile) == "drivers_license"

    prompt = build_system_prompt(profile)

    assert '"next_field_to_collect": "drivers_license"' in prompt
    assert "valid for driving in Spain or Mexico" in prompt
