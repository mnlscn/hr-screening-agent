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
CITY_COORDINATES = {
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3874, 2.1686),
    "Valencia": (39.4699, -0.3763),
    "Sevilla": (37.3891, -5.9845),
    "Malaga": (36.7213, -4.4214),
    "Zaragoza": (41.6488, -0.8891),
    "Bilbao": (43.2630, -2.9350),
    "Ciudad de Mexico": (19.4326, -99.1332),
    "Guadalajara": (20.6597, -103.3496),
    "Monterrey": (25.6866, -100.3161),
    "Puebla": (19.0414, -98.2063),
    "Queretaro": (20.5888, -100.3899),
    "Merida": (20.9674, -89.5926),
}
STALE_ACTIVE_AFTER = timedelta(hours=24)
MAP_BUBBLE_SIZE_SCALE = 35000
