import unittest

from screening.models import CandidateProfile, DeliveryExperience
from screening.prompt import OPENING_MESSAGE, build_system_prompt, get_next_field


class PromptTest(unittest.TestCase):
    def test_opening_message_matches_recruiter_outreach_flow(self):
        self.assertIn("solicitud", OPENING_MESSAGE)
        self.assertIn("conductor/a de reparto", OPENING_MESSAGE)
        self.assertIn("asistente de IA", OPENING_MESSAGE)
        self.assertEqual(OPENING_MESSAGE.count("?"), 1)

    def test_missing_fields_drive_next_question_goal(self):
        profile = CandidateProfile(
            full_name="Giacomo Ortiz",
            drivers_license="Yes",
            raw_city_zone="Guadalajara",
            city_zone="Guadalajara Centro",
            city_zone_status="Matched",
            prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
            start_date="next week",
        )

        self.assertEqual(get_next_field(profile), "availability")

        prompt = build_system_prompt(profile)

        self.assertIn('"next_field_to_collect": "availability"', prompt)
        self.assertIn("Do not ask for phone number, email", prompt)
        self.assertIn("Do not close or wrap up", prompt)

    def test_clarification_fields_are_prioritized(self):
        profile = CandidateProfile(
            full_name="Giacomo Ortiz",
            raw_drivers_license="la mexicana la tendre proxima semana",
            drivers_license="Pending",
            raw_city_zone="Guadalajara",
            city_zone="Guadalajara Centro",
            city_zone_status="Matched",
        )

        self.assertEqual(get_next_field(profile), "drivers_license")

        prompt = build_system_prompt(profile)

        self.assertIn('"next_field_to_collect": "drivers_license"', prompt)
        self.assertIn("valid local driver's license before starting", prompt)


if __name__ == "__main__":
    unittest.main()
