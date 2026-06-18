"""Pure logic for analytics: filtering, KPI computation, and chart data preparation."""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from statistics import mean, median

from screening.domain.models import REQUIRED_FIELDS
from screening.persistence.storage import StoredCandidate
from screening.ui.constants import (
    ANALYTICS_OUTCOME_TITLES,
    CITY_COORDINATES,
    FIELD_TITLES,
    FILTER_ALL,
    MAP_BUBBLE_SIZE_SCALE,
    STALE_ACTIVE_AFTER,
)
from screening.ui.dashboard_logic import candidate_triage
from screening.ui.formatting import (
    candidate_started_date,
    format_percent,
    parse_timestamp,
)


@dataclass(frozen=True)
class AnalyticsFilters:
    """Active filter selections for the analytics tab.

    Attributes:
        start_date (date | None): Inclusive lower bound on the start date, or
            None for no lower bound.
        end_date (date | None): Inclusive upper bound on the start date, or None
            for no upper bound.
        city_zone (str): Selected city, or the "All" sentinel.
        triage (str): Selected analytics outcome title, or the "All" sentinel.
    """

    start_date: date | None
    end_date: date | None
    city_zone: str
    triage: str


@dataclass(frozen=True)
class DurationStats:
    """Aggregate screening-duration statistics.

    Attributes:
        average_minutes (float | None): Mean screening duration in minutes, or
            None when no durations are available.
        median_minutes (float | None): Median screening duration in minutes, or
            None when no durations are available.
    """

    average_minutes: float | None
    median_minutes: float | None


@dataclass(frozen=True)
class AnalyticsKpis:
    """Headline KPIs computed for the analytics tab.

    Attributes:
        started (int): Number of screenings started.
        completed (int): Number of completed screenings.
        completion_rate (float): Completed divided by started, or 0 when none
            started.
        qualified (int): Number of candidates labeled "eligible".
        needs_review (int): Number of candidates triaged as "needs_review".
        average_duration_minutes (float | None): Mean duration in minutes, or
            None when unavailable.
        median_duration_minutes (float | None): Median duration in minutes, or
            None when unavailable.
    """

    started: int
    completed: int
    completion_rate: float
    qualified: int
    needs_review: int
    average_duration_minutes: float | None
    median_duration_minutes: float | None


def ranked_chart_rows(
    rows: list[dict[str, object]],
    *,
    value: str,
    category: str,
) -> list[dict[str, object]]:
    """Sort chart rows by value descending, breaking ties by category.

    Args:
        rows (list[dict[str, object]]): Chart rows to sort.
        value (str): Key holding the numeric value to rank by.
        category (str): Key holding the category label used as a tiebreaker.

    Returns:
        list[dict[str, object]]: The rows sorted by descending value then
            ascending category.
    """
    return sorted(
        rows,
        key=lambda row: (-numeric_row_value(row, value), str(row[category])),
    )


def numeric_row_value(row: dict[str, object], key: str) -> float:
    """Read a numeric value from a chart row, defaulting to 0.

    Args:
        row (dict[str, object]): The chart row.
        key (str): Key to read.

    Returns:
        float: The value as a float, or 0 when missing or non-numeric.
    """
    value = row.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0


def filter_analytics_candidates(
    candidates: list[StoredCandidate],
    filters: AnalyticsFilters,
) -> list[StoredCandidate]:
    """Filter candidates by the analytics filter selections.

    Args:
        candidates (list[StoredCandidate]): Candidates to filter.
        filters (AnalyticsFilters): The active filter selections.

    Returns:
        list[StoredCandidate]: Candidates matching all active filters.
    """
    return [
        candidate
        for candidate in candidates
        if candidate_matches_analytics_filters(candidate, filters)
    ]


def candidate_matches_analytics_filters(
    candidate: StoredCandidate,
    filters: AnalyticsFilters,
) -> bool:
    """Check whether a candidate matches the analytics filters.

    Args:
        candidate (StoredCandidate): The candidate to test.
        filters (AnalyticsFilters): The active filter selections.

    Returns:
        bool: True when the candidate passes the date range, city, and triage
            filters.
    """
    started_date = candidate_started_date(candidate.started_at)
    if filters.start_date is not None:
        if started_date is None or started_date < filters.start_date:
            return False
    if filters.end_date is not None:
        if started_date is None or started_date > filters.end_date:
            return False
    if (
        filters.city_zone != FILTER_ALL
        and candidate.profile.city_zone != filters.city_zone
    ):
        return False
    outcome = analytics_outcome(candidate)
    if (
        filters.triage != FILTER_ALL
        and filters.triage != ANALYTICS_OUTCOME_TITLES[outcome]
    ):
        return False
    return True


def analytics_kpis(candidates: list[StoredCandidate]) -> AnalyticsKpis:
    """Compute headline KPIs for a candidate selection.

    Args:
        candidates (list[StoredCandidate]): Candidates in the current selection.

    Returns:
        AnalyticsKpis: Counts, completion rate, and duration statistics.
    """
    started = len(candidates)
    completed = sum(1 for candidate in candidates if candidate.status == "completed")
    duration_stats = duration_stats_for_candidates(candidates)
    return AnalyticsKpis(
        started=started,
        completed=completed,
        completion_rate=(completed / started) if started else 0,
        qualified=sum(
            1 for candidate in candidates if candidate.bot_label == "eligible"
        ),
        needs_review=sum(
            1
            for candidate in candidates
            if candidate_triage(candidate) == "needs_review"
        ),
        average_duration_minutes=duration_stats.average_minutes,
        median_duration_minutes=duration_stats.median_minutes,
    )


def duration_stats_for_candidates(candidates: list[StoredCandidate]) -> DurationStats:
    """Compute average and median screening duration over candidates.

    Args:
        candidates (list[StoredCandidate]): Candidates to measure.

    Returns:
        DurationStats: Mean and median durations in minutes, both None when no
            candidate has a measurable duration.
    """
    durations = [
        duration
        for duration in (
            candidate_duration_minutes(candidate) for candidate in candidates
        )
        if duration is not None
    ]
    if not durations:
        return DurationStats(average_minutes=None, median_minutes=None)
    return DurationStats(
        average_minutes=mean(durations),
        median_minutes=median(durations),
    )


def candidate_duration_minutes(candidate: StoredCandidate) -> float | None:
    """Compute a candidate's screening duration in minutes.

    Args:
        candidate (StoredCandidate): The candidate to measure.

    Returns:
        float | None: The duration in minutes, or None when the candidate is
            unfinished, has unparseable timestamps, or has a negative duration.
    """
    if candidate.completed_at is None:
        return None
    started_at = parse_timestamp(candidate.started_at)
    completed_at = parse_timestamp(candidate.completed_at)
    if started_at is None or completed_at is None:
        return None
    duration = completed_at - started_at
    if duration.total_seconds() < 0:
        return None
    return duration.total_seconds() / 60


def funnel_completion_rows(
    candidates: list[StoredCandidate],
) -> list[dict[str, object]]:
    """Count how many candidates completed each required field.

    Args:
        candidates (list[StoredCandidate]): Candidates to tally.

    Returns:
        list[dict[str, object]]: One row per required field with its display
            ``stage`` title and a ``completed`` count.
    """
    return [
        {
            "stage": FIELD_TITLES[field],
            "completed": sum(
                1
                for candidate in candidates
                if profile_field_is_complete(candidate, field)
            ),
        }
        for field in REQUIRED_FIELDS
    ]


def funnel_stage_order() -> list[str]:
    """Return funnel stage titles in required-field order.

    Returns:
        list[str]: Display titles for the required fields in canonical order.
    """
    return [FIELD_TITLES[field] for field in REQUIRED_FIELDS]


def profile_field_is_complete(candidate: StoredCandidate, field: str) -> bool:
    """Report whether a single profile field is complete for a candidate.

    Args:
        candidate (StoredCandidate): The candidate to inspect.
        field (str): The field name to check.

    Returns:
        bool: True when the field is neither missing nor awaiting
            clarification.
    """
    profile = candidate.profile
    return (
        field not in profile.missing_fields
        and field not in profile.clarification_fields
    )


def dropoff_stage_rows(candidates: list[StoredCandidate]) -> list[dict[str, object]]:
    """Count candidates by the field where they dropped off.

    Args:
        candidates (list[StoredCandidate]): Candidates to tally.

    Returns:
        list[dict[str, object]]: One row per drop-off field with its display
            ``stage`` title and a ``candidates`` count, sorted by count
            descending.
    """
    counts: dict[str, int] = {}
    for candidate in candidates:
        stage = dropoff_stage(candidate)
        if stage is not None:
            counts[stage] = counts.get(stage, 0) + 1

    return [
        {"stage": FIELD_TITLES.get(stage, stage), "candidates": count}
        for stage, count in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def dropoff_stage(candidate: StoredCandidate) -> str | None:
    """Determine the field where a candidate is currently stalled.

    Args:
        candidate (StoredCandidate): The candidate to inspect.

    Returns:
        str | None: The first clarification or missing field, or None when the
            candidate has finished and has a complete profile.
    """
    if candidate.status != "active" and candidate.profile.is_complete:
        return None
    if candidate.profile.clarification_fields:
        return candidate.profile.clarification_fields[0]
    if candidate.profile.missing_fields:
        return candidate.profile.missing_fields[0]
    return None


def city_distribution_rows(
    candidates: list[StoredCandidate],
) -> list[dict[str, object]]:
    """Build per-city map rows with counts, eligibility, and styling.

    Candidates whose city is unknown or has no coordinates are aggregated into
    a single "Unknown" row without map coordinates.

    Args:
        candidates (list[StoredCandidate]): Candidates to aggregate.

    Returns:
        list[dict[str, object]]: One row per mapped city with count, eligible
            count and share, coordinates, bubble size, and color, plus an
            optional unmapped "Unknown" row.
    """
    city_counts: dict[str, dict[str, int]] = {}
    unknown_count = 0
    for candidate in candidates:
        city = candidate.profile.city_zone
        if city is None or city not in CITY_COORDINATES:
            unknown_count += 1
            continue
        if city not in city_counts:
            city_counts[city] = {"count": 0, "eligible": 0}
        city_counts[city]["count"] += 1
        if candidate.bot_label == "eligible":
            city_counts[city]["eligible"] += 1

    rows: list[dict[str, object]] = []
    for city, counts in sorted(city_counts.items()):
        count = counts["count"]
        eligible_share = counts["eligible"] / count if count else 0
        lat, lon = CITY_COORDINATES[city]
        rows.append(
            {
                "city": city,
                "count": count,
                "eligible_count": counts["eligible"],
                "eligible_share": eligible_share,
                "lat": lat,
                "lon": lon,
                "size": city_bubble_size(count),
                "color": eligible_share_color(eligible_share),
            }
        )

    if unknown_count:
        rows.append(
            {
                "city": "Unknown",
                "count": unknown_count,
                "eligible_count": 0,
                "eligible_share": 0,
                "lat": None,
                "lon": None,
                "size": 0,
                "color": "#9CA3AF",
            }
        )

    return rows


def city_summary_rows(
    distribution_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build a display-ready city summary table from distribution rows.

    Args:
        distribution_rows (list[dict[str, object]]): Rows from
            :func:`city_distribution_rows`.

    Returns:
        list[dict[str, object]]: Rows with title-cased columns for city,
            candidate count, eligible count, and formatted eligible share,
            sorted by count descending.
    """
    return [
        {
            "City": str(row["city"]),
            "Candidates": row["count"],
            "Eligible": row["eligible_count"],
            "Eligible share": format_percent(numeric_row_value(row, "eligible_share")),
        }
        for row in sorted(
            distribution_rows,
            key=lambda row: (-numeric_row_value(row, "count"), str(row["city"])),
        )
    ]


def city_bubble_size(count: int) -> int:
    """Compute a map bubble size for a city candidate count.

    Args:
        count (int): Number of candidates in the city.

    Returns:
        int: The scaled bubble size, never below the base scale.
    """
    return max(MAP_BUBBLE_SIZE_SCALE, count * MAP_BUBBLE_SIZE_SCALE)


def eligible_share_color(eligible_share: float) -> str:
    """Map an eligible share to a bubble color.

    Args:
        eligible_share (float): Fraction of eligible candidates in the city.

    Returns:
        str: Green for shares of 0.5 or more, amber for any positive share, red
            for zero.
    """
    if eligible_share >= 0.5:
        return "#16A34A"
    if eligible_share > 0:
        return "#F59E0B"
    return "#EF4444"


def analytics_outcome_rows(
    candidates: list[StoredCandidate],
) -> list[dict[str, object]]:
    """Count candidates per analytics outcome, omitting empty outcomes.

    Args:
        candidates (list[StoredCandidate]): Candidates to tally.

    Returns:
        list[dict[str, object]]: One row per non-empty outcome with its display
            ``outcome`` title and ``count``.
    """
    counts = {outcome: 0 for outcome in ANALYTICS_OUTCOME_TITLES}
    for candidate in candidates:
        counts[analytics_outcome(candidate)] += 1
    return [
        {"outcome": title, "count": counts[outcome]}
        for outcome, title in ANALYTICS_OUTCOME_TITLES.items()
        if counts[outcome] > 0
    ]


def analytics_outcome(candidate: StoredCandidate) -> str:
    """Classify a candidate into an analytics outcome bucket.

    Args:
        candidate (StoredCandidate): The candidate to classify.

    Returns:
        str: "active_without_summary" for active unsummarized candidates, the
            bot label when "eligible" or "not_eligible", otherwise
            "needs_review".
    """
    if candidate.status == "active" and candidate.bot_label is None:
        return "active_without_summary"
    if candidate.bot_label in {"eligible", "not_eligible"}:
        return candidate.bot_label
    return "needs_review"


def availability_rows(candidates: list[StoredCandidate]) -> list[dict[str, object]]:
    """Count availability preferences among completed or eligible candidates.

    Args:
        candidates (list[StoredCandidate]): Candidates to tally.

    Returns:
        list[dict[str, object]]: One row per availability value with its
            ``availability`` label and ``count``, sorted by count descending.
    """
    counts: dict[str, int] = {}
    for candidate in candidates:
        if candidate.status != "completed" and candidate.bot_label != "eligible":
            continue
        availability = candidate.profile.availability
        if availability is None:
            continue
        counts[availability] = counts.get(availability, 0) + 1

    return [
        {"availability": availability, "count": count}
        for availability, count in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def stale_active_candidates(
    candidates: list[StoredCandidate],
    *,
    now: datetime | None = None,
) -> list[StoredCandidate]:
    """Return active candidates not updated within the staleness window.

    Args:
        candidates (list[StoredCandidate]): Candidates to inspect.
        now (datetime | None): Reference time; defaults to the current UTC time.

    Returns:
        list[StoredCandidate]: Active candidates whose last update is older than
            ``STALE_ACTIVE_AFTER``.
    """
    now = now or datetime.now(UTC)
    stale_candidates = []
    for candidate in candidates:
        if candidate.status != "active":
            continue
        updated_at = parse_timestamp(candidate.updated_at)
        if updated_at is None:
            continue
        if now - updated_at > STALE_ACTIVE_AFTER:
            stale_candidates.append(candidate)
    return stale_candidates
