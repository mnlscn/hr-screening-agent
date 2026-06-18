from typing import Any, cast

import pytest
from pydantic import ValidationError

from screening.domain.models import CandidateProfile, DeliveryExperience


def unchecked(value: object) -> Any:
    return cast(Any, value)


def test_valid_complete_profile():
    profile = CandidateProfile(
        full_name="Maria Garcia",
        drivers_license=unchecked("yes"),
        raw_city_zone="Madrid",
        city_zone="Madrid",
        city_zone_status="Matched",
        availability=unchecked("full time"),
        preferred_schedule=unchecked("morning"),
        prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
        start_date="next Monday",
    )

    assert profile.drivers_license == "Yes"
    assert profile.availability == "Full-time"
    assert profile.preferred_schedule == "Morning"
    assert profile.city_zone_status == "Matched"
    assert profile.is_complete
    assert not profile.is_disqualified
    assert profile.missing_fields == []
    assert profile.clarification_fields == []


def test_invalid_availability_is_rejected():
    with pytest.raises(ValidationError):
        CandidateProfile(availability=unchecked("nights only"))


def test_invalid_preferred_schedule_is_rejected():
    with pytest.raises(ValidationError):
        CandidateProfile(preferred_schedule=unchecked("lunch"))


def test_driver_license_no_disqualifies_internally():
    profile = CandidateProfile(drivers_license="No")

    assert profile.is_disqualified
    assert "driver_license_no" in profile.disqualification_reasons


def test_driver_license_nope_disqualifies_internally():
    profile = CandidateProfile(drivers_license=unchecked("nope"))

    assert profile.drivers_license == "No"
    assert profile.is_disqualified
    assert "driver_license_no" in profile.disqualification_reasons


def test_driver_license_pending_needs_clarification():
    profile = CandidateProfile(
        raw_drivers_license="I'm taking it next week",
        drivers_license=unchecked("I'm taking it next week"),
    )

    assert profile.drivers_license == "Pending"
    assert not profile.is_disqualified
    assert "drivers_license" in profile.clarification_fields


def test_driver_license_natural_yes_is_normalized():
    profile = CandidateProfile(drivers_license=unchecked("I have one"))

    assert profile.drivers_license == "Yes"
    assert not profile.is_disqualified


def test_llm_normalized_barcelona_is_eligible():
    profile = CandidateProfile(
        raw_city_zone="Barcelona",
        city_zone="Barcelona",
        city_zone_status="Matched",
    )

    assert not profile.is_disqualified
    assert profile.city_zone == "Barcelona"


def test_llm_normalized_bcn_is_eligible():
    profile = CandidateProfile(
        raw_city_zone="BCN",
        city_zone="Barcelona",
        city_zone_status="Matched",
    )

    assert not profile.is_disqualified
    assert profile.city_zone == "Barcelona"


def test_llm_normalized_typo_is_eligible():
    profile = CandidateProfile(
        raw_city_zone="barcelna",
        city_zone="Barcelona",
        city_zone_status="Matched",
    )

    assert not profile.is_disqualified
    assert profile.city_zone == "Barcelona"


def test_ambiguous_city_needs_clarification():
    profile = CandidateProfile(
        raw_city_zone="centro",
        city_zone=None,
        city_zone_status="Needs clarification",
    )

    assert not profile.is_disqualified
    assert "city_zone" in profile.clarification_fields


def test_unsupported_city_disqualifies_internally():
    profile = CandidateProfile(
        raw_city_zone="Paris",
        city_zone=None,
        city_zone_status="Unsupported",
    )

    assert profile.is_disqualified
    assert "outside_service_area" in profile.disqualification_reasons


def test_zone_level_city_zone_is_rejected():
    with pytest.raises(ValidationError):
        CandidateProfile(city_zone="Barcelona Eixample")


def test_partial_profile_tracks_missing_fields():
    profile = CandidateProfile(full_name="Luis Perez")

    assert not profile.is_complete
    assert "full_name" not in profile.missing_fields
    assert "drivers_license" in profile.missing_fields


def test_single_name_keeps_full_name_missing():
    profile = CandidateProfile(full_name="Marco")

    assert "full_name" in profile.missing_fields


def test_no_delivery_experience_is_complete_experience_answer():
    profile = CandidateProfile(
        prior_delivery_experience=DeliveryExperience(years=0, platform=None)
    )

    assert profile.prior_delivery_experience is not None
    assert profile.prior_delivery_experience.is_complete()
    assert "prior_delivery_experience" not in profile.missing_fields


def test_no_delivery_experience_text_normalizes_to_zero_years():
    experience = DeliveryExperience(years=unchecked("no experience"), platform=None)

    assert experience.years == 0
    assert experience.is_complete()


def test_nope_delivery_experience_text_normalizes_to_zero_years():
    experience = DeliveryExperience(years=unchecked("nope"), platform=None)

    assert experience.years == 0
    assert experience.is_complete()


@pytest.mark.parametrize(
    "conversation_language",
    ["English", "Spanish", "Mixed"],
)
def test_conversation_language_accepts_supported_values(conversation_language: str):
    profile = CandidateProfile(conversation_language=unchecked(conversation_language))

    assert profile.conversation_language == conversation_language


@pytest.mark.parametrize(
    "conversation_language",
    ["eng", "esp", "bilingual", "it"],
)
def test_conversation_language_rejects_unclear_labels(conversation_language: str):
    with pytest.raises(ValidationError):
        CandidateProfile(conversation_language=unchecked(conversation_language))


def test_merge_updates_conversation_language():
    profile = CandidateProfile(conversation_language="Spanish")
    updates = CandidateProfile(conversation_language="Mixed")

    merged = profile.merge(updates)

    assert merged.conversation_language == "Mixed"
