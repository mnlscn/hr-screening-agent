"""Display formatting helpers for values, timestamps, durations, and profiles."""

from collections.abc import Iterable
from datetime import UTC, date, datetime

from anthropic.types import MessageParam

from screening.domain.models import CandidateProfile, DeliveryExperience


def format_experience(experience: DeliveryExperience | None) -> str:
    """Format delivery experience as a human-readable string.

    Args:
        experience (DeliveryExperience | None): The experience to format.

    Returns:
        str: A comma-separated "years" and platform string, or "-" when there
            is no experience data.
    """
    if experience is None:
        return "-"

    values = []
    if experience.years is not None:
        values.append(f"{experience.years:g} years")
    if experience.platform is not None:
        values.append(experience.platform)

    return ", ".join(values) if values else "-"


def format_value(value: object) -> str:
    """Format an arbitrary value as display text, defaulting empties to "-".

    Args:
        value (object): The value to format.

    Returns:
        str: The stripped string form of the value, or "-" when None or empty.
    """
    if value is None:
        return "-"
    value_text = str(value).strip()
    return value_text if value_text else "-"


def format_list(values: Iterable[str]) -> str:
    """Join non-empty values into a comma-separated string.

    Args:
        values (Iterable[str]): The values to join.

    Returns:
        str: The comma-separated non-empty values, or "-" when none remain.
    """
    formatted_values = [value for value in values if value]
    return ", ".join(formatted_values) if formatted_values else "-"


def format_timestamp(value: str | None) -> str:
    """Format an ISO timestamp string as "YYYY-MM-DD HH:MM".

    Args:
        value (str | None): The ISO timestamp to format.

    Returns:
        str: The formatted timestamp, the raw value when it cannot be parsed,
            or "-" when None.
    """
    if value is None:
        return "-"
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return format_value(value)
    return timestamp.strftime("%Y-%m-%d %H:%M")


def candidate_started_date(started_at: str | None) -> date | None:
    """Extract the calendar date from a candidate's start timestamp.

    Args:
        started_at (str | None): The ISO start timestamp.

    Returns:
        date | None: The UTC date, or None when the timestamp is missing or
            unparseable.
    """
    timestamp = parse_timestamp(started_at)
    return timestamp.date() if timestamp is not None else None


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO timestamp into a timezone-aware UTC datetime.

    Naive timestamps are assumed to be UTC; aware ones are converted to UTC.

    Args:
        value (str | None): The ISO timestamp to parse.

    Returns:
        datetime | None: The UTC datetime, or None when missing or unparseable.
    """
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
    """Format a fraction as a whole-number percentage string.

    Args:
        value (float): The fraction to format (e.g. 0.5).

    Returns:
        str: The value rendered as a percentage with no decimals (e.g. "50%").
    """
    return f"{value:.0%}"


def format_duration(minutes: float | None) -> str:
    """Format a duration in minutes as minutes or hours.

    Args:
        minutes (float | None): The duration in minutes.

    Returns:
        str: A minutes string under one hour, an hours string otherwise, or
            "-" when None.
    """
    if minutes is None:
        return "-"
    if minutes < 60:
        return f"{minutes:.0f} min"
    hours = minutes / 60
    return f"{hours:.1f} h"


def message_text(message: MessageParam) -> str:
    """Return a message's content as plain text.

    Args:
        message (MessageParam): The message to read.

    Returns:
        str: The string content, or its ``str()`` form when not a plain string.
    """
    content = message["content"]
    if isinstance(content, str):
        return content
    return str(content)


def profile_rows(profile: CandidateProfile) -> list[tuple[str, str]]:
    """Build labeled display rows for a candidate profile.

    Args:
        profile (CandidateProfile): The profile to render.

    Returns:
        list[tuple[str, str]]: ``(label, formatted value)`` pairs for the
            profile's display fields.
    """
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
    """Abbreviate a candidate identifier for compact display.

    Args:
        candidate_id (str): The full candidate identifier.

    Returns:
        str: The identifier unchanged when 12 characters or fewer, otherwise an
            elided "prefix...suffix" form.
    """
    if len(candidate_id) <= 12:
        return candidate_id
    return f"{candidate_id[:8]}...{candidate_id[-4:]}"
