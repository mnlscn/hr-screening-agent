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
    start_date: date | None
    end_date: date | None
    city_zone: str
    triage: str


@dataclass(frozen=True)
class DurationStats:
    average_minutes: float | None
    median_minutes: float | None


@dataclass(frozen=True)
class AnalyticsKpis:
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
    return sorted(
        rows,
        key=lambda row: (-numeric_row_value(row, value), str(row[category])),
    )


def numeric_row_value(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0


def filter_analytics_candidates(
    candidates: list[StoredCandidate],
    filters: AnalyticsFilters,
) -> list[StoredCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate_matches_analytics_filters(candidate, filters)
    ]


def candidate_matches_analytics_filters(
    candidate: StoredCandidate,
    filters: AnalyticsFilters,
) -> bool:
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
    return [FIELD_TITLES[field] for field in REQUIRED_FIELDS]


def profile_field_is_complete(candidate: StoredCandidate, field: str) -> bool:
    profile = candidate.profile
    return (
        field not in profile.missing_fields
        and field not in profile.clarification_fields
    )


def dropoff_stage_rows(candidates: list[StoredCandidate]) -> list[dict[str, object]]:
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
    return max(MAP_BUBBLE_SIZE_SCALE, count * MAP_BUBBLE_SIZE_SCALE)


def eligible_share_color(eligible_share: float) -> str:
    if eligible_share >= 0.5:
        return "#16A34A"
    if eligible_share > 0:
        return "#F59E0B"
    return "#EF4444"


def analytics_outcome_rows(
    candidates: list[StoredCandidate],
) -> list[dict[str, object]]:
    counts = {outcome: 0 for outcome in ANALYTICS_OUTCOME_TITLES}
    for candidate in candidates:
        counts[analytics_outcome(candidate)] += 1
    return [
        {"outcome": title, "count": counts[outcome]}
        for outcome, title in ANALYTICS_OUTCOME_TITLES.items()
        if counts[outcome] > 0
    ]


def analytics_outcome(candidate: StoredCandidate) -> str:
    if candidate.status == "active" and candidate.bot_label is None:
        return "active_without_summary"
    if candidate.bot_label in {"eligible", "not_eligible"}:
        return candidate.bot_label
    return "needs_review"


def availability_rows(candidates: list[StoredCandidate]) -> list[dict[str, object]]:
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
