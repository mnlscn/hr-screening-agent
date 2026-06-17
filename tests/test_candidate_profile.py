from typing import Any, cast

import pytest
from pydantic import ValidationError

from screening.models import CandidateProfile, DeliveryExperience


def unchecked(value: object) -> Any:
    return cast(Any, value)


def test_valid_complete_profile():
    profile = CandidateProfile(
        full_name="Maria Garcia",
        drivers_license=unchecked("yes"),
        raw_city_zone="Madrid",
        city_zone="Madrid Centro",
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
        city_zone="Barcelona Eixample",
        city_zone_status="Matched",
    )

    assert not profile.is_disqualified
    assert profile.city_zone == "Barcelona Eixample"


def test_llm_normalized_bcn_is_eligible():
    profile = CandidateProfile(
        raw_city_zone="BCN",
        city_zone="Barcelona Eixample",
        city_zone_status="Matched",
    )

    assert not profile.is_disqualified
    assert profile.city_zone == "Barcelona Eixample"


def test_llm_normalized_typo_is_eligible():
    profile = CandidateProfile(
        raw_city_zone="barcelna",
        city_zone="Barcelona Eixample",
        city_zone_status="Matched",
    )

    assert not profile.is_disqualified
    assert profile.city_zone == "Barcelona Eixample"


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


def test_non_canonical_city_zone_is_rejected():
    with pytest.raises(ValidationError):
        CandidateProfile(city_zone="Barcelona")


def test_partial_profile_tracks_missing_fields():
    profile = CandidateProfile(full_name="Luis Perez")

    assert not profile.is_complete
    assert "full_name" not in profile.missing_fields
    assert "drivers_license" in profile.missing_fields
