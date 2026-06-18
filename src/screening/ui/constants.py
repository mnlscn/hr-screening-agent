"""UI constants: session keys, display labels, and thresholds."""

from datetime import timedelta
from typing import get_args

from screening.domain.models import BotLabel

AGENT_SESSION_KEY = "screening_agent"
CANDIDATE_ID_PARAM = "candidate_id"
CHAT_WINDOW_HEIGHT = 600
FILTER_ALL = "All"
TRIAGE_LABELS = get_args(BotLabel)
TRIAGE_TITLES = {
    "eligible": "Eligible",
    "not_eligible": "Not eligible",
    "needs_review": "Needs review",
}
ANALYTICS_OUTCOME_TITLES = {
    "eligible": "Eligible",
    "not_eligible": "Not eligible",
    "needs_review": "Needs review",
    "active_without_summary": "Active without summary",
}
FIELD_TITLES = {
    "full_name": "Full name",
    "drivers_license": "Driver license",
    "city_zone": "City / zone",
    "availability": "Availability",
    "preferred_schedule": "Schedule",
    "prior_delivery_experience": "Delivery experience",
    "start_date": "Start date",
}
STALE_ACTIVE_AFTER = timedelta(hours=24)
MAP_BUBBLE_SIZE_SCALE = 35000
