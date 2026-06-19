"""Tests for CandidateProfile validation and merge logic."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from screening.domain.models import CandidateProfile, DeliveryExperience


def unchecked(value: object) -> Any:
    return cast(Any, value)


def test_valid_complete_profile():
    profile = CandidateProfile(
        full_name="Maria Garcia",
        drivers_license="Yes",
        raw_city_zone="Madrid",
        city_zone="Madrid",
        city_zone_status="Matched",
        availability="Full-time",
        preferred_schedule="Morning",
        prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
        start_date="next Monday",
    )

    assert profile.drivers_license == "Yes"
    assert profile.availability == "Full-time"
    assert profile.availability_status == "Matched"
    assert profile.preferred_schedule == "Morning"
    assert profile.preferred_schedule_status == "Matched"
    assert profile.city_zone_status == "Matched"
    assert profile.full_name_status == "Matched"
    assert profile.prior_delivery_experience_status == "Matched"
    assert profile.start_date_status == "Matched"
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


def test_driver_license_pending_needs_clarification():
    profile = CandidateProfile(
        raw_drivers_license="I'm taking it next week",
        drivers_license="Pending",
    )

    assert profile.drivers_license == "Pending"
    assert not profile.is_disqualified
    assert "drivers_license" in profile.clarification_fields


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
    assert "city_zone" not in profile.missing_fields
    assert "city_zone" in profile.clarification_fields


def test_unsupported_city_disqualifies_internally():
    profile = CandidateProfile(
        raw_city_zone="Paris",
        city_zone=None,
        city_zone_status="Unsupported",
    )

    assert profile.is_disqualified
    assert "city_zone" not in profile.missing_fields
    assert "outside_service_area" in profile.disqualification_reasons


def test_zone_level_city_zone_is_rejected():
    with pytest.raises(ValidationError):
        CandidateProfile(city_zone="Barcelona Eixample")


def test_partial_profile_tracks_missing_fields():
    profile = CandidateProfile(full_name="Luis Perez")

    assert not profile.is_complete
    assert "full_name" not in profile.missing_fields
    assert "drivers_license" in profile.missing_fields


def test_single_name_needs_clarification_not_missing():
    profile = CandidateProfile(full_name="Marco")

    assert profile.full_name_status == "Needs clarification"
    assert "full_name" not in profile.missing_fields
    assert "full_name" in profile.clarification_fields


def test_no_delivery_experience_is_complete_experience_answer():
    profile = CandidateProfile(
        prior_delivery_experience=DeliveryExperience(years=0, platform=None)
    )

    assert profile.prior_delivery_experience is not None
    assert profile.prior_delivery_experience_status == "Matched"
    assert profile.prior_delivery_experience.is_complete()
    assert "prior_delivery_experience" not in profile.missing_fields


def test_raw_ambiguous_availability_needs_clarification_not_missing():
    profile = CandidateProfile(raw_availability="depende de los dias")

    assert profile.availability is None
    assert profile.availability_status == "Needs clarification"
    assert "availability" not in profile.missing_fields
    assert "availability" in profile.clarification_fields


def test_raw_ambiguous_schedule_needs_clarification_not_missing():
    profile = CandidateProfile(raw_preferred_schedule="a la hora de comida")

    assert profile.preferred_schedule is None
    assert profile.preferred_schedule_status == "Needs clarification"
    assert "preferred_schedule" not in profile.missing_fields
    assert "preferred_schedule" in profile.clarification_fields


def test_platform_only_experience_needs_clarification_not_missing():
    profile = CandidateProfile(
        raw_prior_delivery_experience="Glovo",
        prior_delivery_experience=DeliveryExperience(platform="Glovo"),
    )

    assert profile.prior_delivery_experience_status == "Needs clarification"
    assert "prior_delivery_experience" not in profile.missing_fields
    assert "prior_delivery_experience" in profile.clarification_fields


def test_uncertain_start_date_needs_clarification_not_missing():
    profile = CandidateProfile(raw_start_date="no se, depende")

    assert profile.start_date is None
    assert profile.start_date_status == "Needs clarification"
    assert "start_date" not in profile.missing_fields
    assert "start_date" in profile.clarification_fields


def test_ambiguous_update_clears_existing_canonical_value():
    profile = CandidateProfile(availability="Full-time")
    updates = CandidateProfile(
        raw_availability="depende de la semana",
        availability_status="Needs clarification",
    )

    merged = profile.merge(updates)

    assert merged.availability is None
    assert merged.raw_availability == "depende de la semana"
    assert merged.availability_status == "Needs clarification"
    assert "availability" in merged.clarification_fields


def test_unrelated_null_update_preserves_existing_canonical_value():
    profile = CandidateProfile(availability="Weekends")

    merged = profile.merge(CandidateProfile())

    assert merged.availability == "Weekends"
    assert merged.availability_status == "Matched"


def test_delivery_experience_extra_keys_are_rejected():
    with pytest.raises(ValidationError):
        DeliveryExperience.model_validate(
            {"years": 1, "platform": "Glovo", "notes": "bike courier"}
        )


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
