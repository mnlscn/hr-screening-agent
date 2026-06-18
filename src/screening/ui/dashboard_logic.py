"""Pure logic for the HR dashboard: filtering, grouping, triage, and card data."""

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
    """Active filter selections for the HR dashboard.

    Attributes:
        triage (str): Selected triage title, or the "All" sentinel.
        status (str): Selected lifecycle status, or the "All" sentinel.
        city_zone (str): Selected city, or the "All" sentinel.
        search (str): Free-text search query.
    """

    triage: str
    status: str
    city_zone: str
    search: str


def filter_candidates(
    candidates: list[StoredCandidate],
    filters: DashboardFilters,
) -> list[StoredCandidate]:
    """Filter candidates by the dashboard filter selections.

    Args:
        candidates (list[StoredCandidate]): Candidates to filter.
        filters (DashboardFilters): The active filter selections.

    Returns:
        list[StoredCandidate]: Candidates matching all active filters.
    """
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
    """Check whether a candidate matches the given dashboard filters.

    Args:
        candidate (StoredCandidate): The candidate to test.
        filters (DashboardFilters): The active filter selections.
        normalized_search (str): The lowercased, stripped search query.

    Returns:
        bool: True when the candidate passes the triage, status, city, and
            search filters.
    """
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
    """Group candidates into lists keyed by triage label.

    Args:
        candidates (list[StoredCandidate]): Candidates to group.

    Returns:
        dict[str, list[StoredCandidate]]: Each triage label mapped to its
            candidates; every known label is present, possibly empty.
    """
    grouped = {triage_label: [] for triage_label in TRIAGE_LABELS}
    for candidate in candidates:
        grouped[candidate_triage(candidate)].append(candidate)
    return grouped


def triage_counts(candidates: list[StoredCandidate]) -> dict[str, int]:
    """Count candidates per triage label.

    Args:
        candidates (list[StoredCandidate]): Candidates to count.

    Returns:
        dict[str, int]: Each triage label mapped to its candidate count; every
            known label is present.
    """
    counts = {triage_label: 0 for triage_label in TRIAGE_LABELS}
    for candidate in candidates:
        counts[candidate_triage(candidate)] += 1
    return counts


def candidate_triage(candidate: StoredCandidate) -> str:
    """Return a candidate's triage label, defaulting to "needs_review".

    Args:
        candidate (StoredCandidate): The candidate to classify.

    Returns:
        str: The candidate's bot label when it is a valid triage label,
            otherwise "needs_review".
    """
    if candidate.bot_label in TRIAGE_LABELS:
        return candidate.bot_label
    return "needs_review"


def candidate_display_name(candidate: StoredCandidate) -> str:
    """Return a display name for a candidate.

    Args:
        candidate (StoredCandidate): The candidate to name.

    Returns:
        str: The candidate's full name, or a "Candidate <short id>" fallback
            when no name is available.
    """
    return format_value(candidate.profile.full_name).replace("-", "") or (
        f"Candidate {short_candidate_id(candidate.id)}"
    )


def candidate_card_label(candidate: StoredCandidate) -> str:
    """Build the one-line label shown on a candidate's dashboard card.

    Args:
        candidate (StoredCandidate): The candidate to label.

    Returns:
        str: A dot-separated label with name, triage, short id, and last update.
    """
    return (
        f"{candidate_display_name(candidate)} · "
        f"{TRIAGE_TITLES[candidate_triage(candidate)]} · "
        f"{short_candidate_id(candidate.id)} · "
        f"Updated {format_timestamp(candidate.updated_at)}"
    )


def candidate_requirement_rows(candidate: StoredCandidate) -> list[tuple[str, str]]:
    """Build labeled requirement rows for a candidate card.

    Args:
        candidate (StoredCandidate): The candidate to summarize.

    Returns:
        list[tuple[str, str]]: ``(label, value)`` pairs covering status,
            license, city, availability, schedule, experience, start date,
            summary status, and triage label.
    """
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
