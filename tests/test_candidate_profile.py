import unittest

from pydantic import ValidationError

from screening.models import CandidateProfile, DeliveryExperience


class CandidateProfileTest(unittest.TestCase):
    def test_valid_complete_profile(self):
        profile = CandidateProfile(
            full_name="Maria Garcia",
            drivers_license="yes",
            city_zone="Madrid Centro",
            availability="full time",
            preferred_schedule="morning",
            prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
            start_date="next Monday",
        )

        self.assertEqual(profile.drivers_license, "Yes")
        self.assertEqual(profile.availability, "Full-time")
        self.assertEqual(profile.preferred_schedule, "Morning")
        self.assertTrue(profile.is_complete)
        self.assertFalse(profile.is_disqualified)
        self.assertEqual(profile.missing_fields, [])

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

    def test_outside_service_area_disqualifies_internally(self):
        profile = CandidateProfile(city_zone="Paris Centro")

        self.assertTrue(profile.is_disqualified)
        self.assertIn("outside_service_area", profile.disqualification_reasons)

    def test_partial_profile_tracks_missing_fields(self):
        profile = CandidateProfile(full_name="Luis Perez")

        self.assertFalse(profile.is_complete)
        self.assertNotIn("full_name", profile.missing_fields)
        self.assertIn("drivers_license", profile.missing_fields)


if __name__ == "__main__":
    unittest.main()

