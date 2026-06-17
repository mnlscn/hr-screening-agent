import unittest

from pydantic import ValidationError

from screening.models import CandidateProfile, DeliveryExperience


class CandidateProfileTest(unittest.TestCase):
    def test_valid_complete_profile(self):
        profile = CandidateProfile(
            full_name="Maria Garcia",
            drivers_license="yes",
            raw_city_zone="Madrid",
            city_zone="Madrid Centro",
            city_zone_status="Matched",
            availability="full time",
            preferred_schedule="morning",
            prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
            start_date="next Monday",
        )

        self.assertEqual(profile.drivers_license, "Yes")
        self.assertEqual(profile.availability, "Full-time")
        self.assertEqual(profile.preferred_schedule, "Morning")
        self.assertEqual(profile.city_zone_status, "Matched")
        self.assertTrue(profile.is_complete)
        self.assertFalse(profile.is_disqualified)
        self.assertEqual(profile.missing_fields, [])
        self.assertEqual(profile.clarification_fields, [])

    def test_invalid_availability_is_rejected(self):
        with self.assertRaises(ValidationError):
            CandidateProfile(availability="nights only")

    def test_invalid_preferred_schedule_is_rejected(self):
        with self.assertRaises(ValidationError):
            CandidateProfile(preferred_schedule="lunch")

    def test_driver_license_no_disqualifies_internally(self):
        profile = CandidateProfile(drivers_license="No")

        self.assertTrue(profile.is_disqualified)
        self.assertIn("driver_license_no", profile.disqualification_reasons)

    def test_driver_license_pending_needs_clarification(self):
        profile = CandidateProfile(
            raw_drivers_license="I'm taking it next week",
            drivers_license="I'm taking it next week",
        )

        self.assertEqual(profile.drivers_license, "Pending")
        self.assertFalse(profile.is_disqualified)
        self.assertIn("drivers_license", profile.clarification_fields)

    def test_driver_license_natural_yes_is_normalized(self):
        profile = CandidateProfile(drivers_license="I have one")

        self.assertEqual(profile.drivers_license, "Yes")
        self.assertFalse(profile.is_disqualified)

    def test_llm_normalized_barcelona_is_eligible(self):
        profile = CandidateProfile(
            raw_city_zone="Barcelona",
            city_zone="Barcelona Eixample",
            city_zone_status="Matched",
        )

        self.assertFalse(profile.is_disqualified)
        self.assertEqual(profile.city_zone, "Barcelona Eixample")

    def test_llm_normalized_bcn_is_eligible(self):
        profile = CandidateProfile(
            raw_city_zone="BCN",
            city_zone="Barcelona Eixample",
            city_zone_status="Matched",
        )

        self.assertFalse(profile.is_disqualified)
        self.assertEqual(profile.city_zone, "Barcelona Eixample")

    def test_llm_normalized_typo_is_eligible(self):
        profile = CandidateProfile(
            raw_city_zone="barcelna",
            city_zone="Barcelona Eixample",
            city_zone_status="Matched",
        )

        self.assertFalse(profile.is_disqualified)
        self.assertEqual(profile.city_zone, "Barcelona Eixample")

    def test_ambiguous_city_needs_clarification(self):
        profile = CandidateProfile(
            raw_city_zone="centro",
            city_zone=None,
            city_zone_status="Needs clarification",
        )

        self.assertFalse(profile.is_disqualified)
        self.assertIn("city_zone", profile.clarification_fields)

    def test_unsupported_city_disqualifies_internally(self):
        profile = CandidateProfile(
            raw_city_zone="Paris",
            city_zone=None,
            city_zone_status="Unsupported",
        )

        self.assertTrue(profile.is_disqualified)
        self.assertIn("outside_service_area", profile.disqualification_reasons)

    def test_non_canonical_city_zone_is_rejected(self):
        with self.assertRaises(ValidationError):
            CandidateProfile(city_zone="Barcelona")

    def test_partial_profile_tracks_missing_fields(self):
        profile = CandidateProfile(full_name="Luis Perez")

        self.assertFalse(profile.is_complete)
        self.assertNotIn("full_name", profile.missing_fields)
        self.assertIn("drivers_license", profile.missing_fields)


if __name__ == "__main__":
    unittest.main()
