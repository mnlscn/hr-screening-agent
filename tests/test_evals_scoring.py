"""Tests for extraction-eval scoring and case loading."""

import json

import pytest

from screening.domain.models import CandidateProfile, DeliveryExperience
from screening.evals.scoring import (
    aggregate,
    flatten_expected,
    format_report,
    load_cases,
    score_case,
    score_error,
)


def test_score_case_matches_exact_soft_and_nested_fields():
    profile = CandidateProfile(
        conversation_language="Spanish",
        full_name="Maria Garcia",
        drivers_license="Yes",
        city_zone="Madrid",
        city_zone_status="Matched",
        availability="Full-time",
        preferred_schedule="Morning",
        prior_delivery_experience=DeliveryExperience(years=2.0, platform="GLOVO"),
        start_date="Next Monday",
    )

    result = score_case(
        "happy",
        "multi_field",
        {
            "conversation_language": "Spanish",
            "full_name": " maria garcia ",
            "drivers_license": "Yes",
            "city_zone": "Madrid",
            "city_zone_status": "Matched",
            "availability": "Full-time",
            "availability_status": "Matched",
            "preferred_schedule": "Morning",
            "prior_delivery_experience": {
                "years": 2,
                "platform": "glovo",
            },
            "start_date": "next monday",
        },
        profile,
    )

    assert result.passed
    assert {field.field for field in result.fields if field.soft} == {
        "full_name",
        "prior_delivery_experience.platform",
        "start_date",
    }


def test_score_case_reports_mismatches_for_partial_expected_fields():
    profile = CandidateProfile(
        drivers_license="No",
        raw_city_zone="centro",
        city_zone_status="Needs clarification",
    )

    result = score_case(
        "mismatch",
        "license",
        {
            "drivers_license": "Yes",
            "city_zone_status": "Unsupported",
        },
        profile,
    )

    assert not result.passed
    assert [(field.field, field.expected, field.got) for field in result.fields] == [
        ("drivers_license", "Yes", "No"),
        ("city_zone_status", "Unsupported", "Needs clarification"),
    ]


def test_score_error_counts_exception_as_failed_assertions():
    result = score_error(
        "api-failure",
        "city",
        {"city_zone": "Madrid", "city_zone_status": "Matched"},
        RuntimeError("boom"),
    )

    assert not result.passed
    assert result.error == "RuntimeError: boom"
    assert all(not field.correct for field in result.fields)
    assert {field.error for field in result.fields} == {"RuntimeError: boom"}


def test_aggregate_counts_accuracy_buckets_and_confusion_matrix():
    passed = score_case(
        "passed",
        "license",
        {"drivers_license": "Yes"},
        CandidateProfile(drivers_license="Yes"),
    )
    failed = score_case(
        "failed",
        "license",
        {"drivers_license": "Yes"},
        CandidateProfile(drivers_license="No"),
    )
    city = score_case(
        "city",
        "city",
        {"city_zone_status": "Unsupported"},
        CandidateProfile(
            raw_city_zone="centro", city_zone_status="Needs clarification"
        ),
    )

    summary = aggregate([passed, failed, city])

    assert summary.case_count == 3
    assert summary.passed_cases == 1
    assert summary.overall_accuracy.correct == 1
    assert summary.overall_accuracy.total == 3
    assert summary.field_accuracy["drivers_license"].correct == 1
    assert summary.field_accuracy["drivers_license"].total == 2
    assert summary.bucket_accuracy["license"].correct == 1
    assert summary.bucket_accuracy["license"].total == 2
    assert summary.confusion_matrices["drivers_license"]["Yes"] == {
        "No": 1,
        "Yes": 1,
    }
    assert summary.confusion_matrices["city_zone_status"]["Unsupported"] == {
        "Needs clarification": 1
    }
    assert len(summary.failures) == 2


def test_format_report_includes_key_sections():
    summary = aggregate(
        [
            score_case(
                "passed",
                "license",
                {"drivers_license": "Yes"},
                CandidateProfile(drivers_license="Yes"),
            )
        ]
    )

    report = format_report(summary)

    assert "Extraction Eval Report" in report
    assert "Per-field accuracy" in report
    assert "Per-bucket accuracy" in report
    assert "Confusion matrices" in report
    assert "Failures" in report


def test_flatten_expected_rejects_raw_and_derived_fields():
    with pytest.raises(ValueError, match="raw fields are not scored"):
        flatten_expected({"raw_city_zone": "Madrid"})

    with pytest.raises(ValueError, match="derived fields are not scored"):
        flatten_expected({"missing_fields": []})

    assert flatten_expected({"availability_status": "Needs clarification"}) == {
        "availability_status": "Needs clarification"
    }


def test_load_cases_rejects_malformed_expected_field(tmp_path):
    case_path = tmp_path / "bad_cases.jsonl"
    case_path.write_text(
        json.dumps(
            {
                "id": "bad",
                "bucket": "city",
                "transcript": [{"role": "user", "content": "Madrid"}],
                "expected": {"raw_city_zone": "Madrid"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw fields are not scored"):
        load_cases(case_path)


def test_bundled_cases_are_valid_and_unique():
    cases = load_cases()

    assert 30 <= len(cases) <= 40
    assert len({case.id for case in cases}) == len(cases)
    assert any(case.id == "city-alias-cdmx" for case in cases)
    assert any(case.current_profile.full_name == "Luis Perez" for case in cases)

    for case in cases:
        assert case.id
        assert case.bucket
        assert case.transcript
        assert case.current_profile
        assert flatten_expected(case.expected)
