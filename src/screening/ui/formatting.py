from collections.abc import Iterable
from datetime import UTC, date, datetime

from anthropic.types import MessageParam

from screening.models import CandidateProfile, DeliveryExperience


def format_experience(experience: DeliveryExperience | None) -> str:
    if experience is None:
        return "-"

    values = []
    if experience.years is not None:
        values.append(f"{experience.years:g} years")
    if experience.platform is not None:
        values.append(experience.platform)

    return ", ".join(values) if values else "-"


def format_value(value: object) -> str:
    if value is None:
        return "-"
    value_text = str(value).strip()
    return value_text if value_text else "-"


def format_list(values: Iterable[str]) -> str:
    formatted_values = [value for value in values if value]
    return ", ".join(formatted_values) if formatted_values else "-"


def format_timestamp(value: str | None) -> str:
    if value is None:
        return "-"
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return format_value(value)
    return timestamp.strftime("%Y-%m-%d %H:%M")


def candidate_started_date(started_at: str | None) -> date | None:
    timestamp = parse_timestamp(started_at)
    return timestamp.date() if timestamp is not None else None


def parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def format_percent(value: float) -> str:
    return f"{value:.0%}"


def format_duration(minutes: float | None) -> str:
    if minutes is None:
        return "-"
    if minutes < 60:
        return f"{minutes:.0f} min"
    hours = minutes / 60
    return f"{hours:.1f} h"


def message_text(message: MessageParam) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return str(content)


def profile_rows(profile: CandidateProfile) -> list[tuple[str, str]]:
    return [
        ("Full name", format_value(profile.full_name)),
        ("Drivers license", format_value(profile.drivers_license)),
        ("Raw city or zone", format_value(profile.raw_city_zone)),
        ("City or zone", format_value(profile.city_zone)),
        ("City status", format_value(profile.city_zone_status)),
        ("Language", format_value(profile.conversation_language)),
        ("Availability", format_value(profile.availability)),
        ("Schedule", format_value(profile.preferred_schedule)),
        ("Experience", format_experience(profile.prior_delivery_experience)),
        ("Start date", format_value(profile.start_date)),
    ]


def short_candidate_id(candidate_id: str) -> str:
    if len(candidate_id) <= 12:
        return candidate_id
    return f"{candidate_id[:8]}...{candidate_id[-4:]}"
