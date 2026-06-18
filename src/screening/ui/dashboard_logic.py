from dataclasses import dataclass

from screening.persistence.storage import StoredCandidate
from screening.ui.constants import FILTER_ALL, TRIAGE_LABELS, TRIAGE_TITLES
from screening.ui.formatting import (
    format_experience,
    format_value,
    format_timestamp,
    short_candidate_id,
)


@dataclass(frozen=True)
class DashboardFilters:
    triage: str
    status: str
    city_zone: str
    search: str


def filter_candidates(
    candidates: list[StoredCandidate],
    filters: DashboardFilters,
) -> list[StoredCandidate]:
    normalized_search = filters.search.strip().lower()
    return [
        candidate
        for candidate in candidates
        if candidate_matches_filters(candidate, filters, normalized_search)
    ]


def candidate_matches_filters(
    candidate: StoredCandidate,
    filters: DashboardFilters,
    normalized_search: str,
) -> bool:
    triage = candidate_triage(candidate)
    if filters.triage != FILTER_ALL and filters.triage != TRIAGE_TITLES[triage]:
        return False
    if filters.status != FILTER_ALL and filters.status != candidate.status:
        return False
    if (
        filters.city_zone != FILTER_ALL
        and filters.city_zone != candidate.profile.city_zone
    ):
        return False
    if not normalized_search:
        return True

    searchable_text = " ".join(
        (
            candidate.id,
            candidate_display_name(candidate),
            format_value(candidate.profile.city_zone),
            format_value(candidate.hr_summary),
        )
    ).lower()
    return normalized_search in searchable_text


def group_candidates_by_triage(
    candidates: list[StoredCandidate],
) -> dict[str, list[StoredCandidate]]:
    grouped = {triage_label: [] for triage_label in TRIAGE_LABELS}
    for candidate in candidates:
        grouped[candidate_triage(candidate)].append(candidate)
    return grouped


def triage_counts(candidates: list[StoredCandidate]) -> dict[str, int]:
    counts = {triage_label: 0 for triage_label in TRIAGE_LABELS}
    for candidate in candidates:
        counts[candidate_triage(candidate)] += 1
    return counts


def candidate_triage(candidate: StoredCandidate) -> str:
    if candidate.bot_label in TRIAGE_LABELS:
        return candidate.bot_label
    return "needs_review"


def candidate_display_name(candidate: StoredCandidate) -> str:
    return format_value(candidate.profile.full_name).replace("-", "") or (
        f"Candidate {short_candidate_id(candidate.id)}"
    )


def candidate_card_label(candidate: StoredCandidate) -> str:
    return (
        f"{candidate_display_name(candidate)} · "
        f"{TRIAGE_TITLES[candidate_triage(candidate)]} · "
        f"{short_candidate_id(candidate.id)} · "
        f"Updated {format_timestamp(candidate.updated_at)}"
    )


def candidate_requirement_rows(candidate: StoredCandidate) -> list[tuple[str, str]]:
    profile = candidate.profile
    return [
        ("Status", format_value(candidate.status)),
        ("License", format_value(profile.drivers_license)),
        ("City", format_value(profile.city_zone)),
        ("Availability", format_value(profile.availability)),
        ("Schedule", format_value(profile.preferred_schedule)),
        ("Experience", format_experience(profile.prior_delivery_experience)),
        ("Start date", format_value(profile.start_date)),
        ("Summary", format_value(candidate.summary_status)),
        ("Label", TRIAGE_TITLES[candidate_triage(candidate)]),
    ]
