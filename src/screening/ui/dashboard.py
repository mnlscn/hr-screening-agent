import streamlit as st

from screening.config import SCREENING_DB_PATH
from screening.persistence.storage import StoredCandidate, list_candidates
from screening.ui.constants import FILTER_ALL, TRIAGE_LABELS, TRIAGE_TITLES
from screening.ui.dashboard_logic import (
    DashboardFilters,
    candidate_card_label,
    candidate_requirement_rows,
    candidate_triage,
    filter_candidates,
    group_candidates_by_triage,
    triage_counts,
)
from screening.ui.formatting import format_list
from screening.ui.state import open_candidate_chat


def render_dashboard() -> None:
    candidates = list_candidates(db_path=SCREENING_DB_PATH)
    render_dashboard_metrics(candidates)

    if not candidates:
        st.info("No candidates in the database yet.")
        return

    filters = render_dashboard_filters(candidates)
    filtered_candidates = filter_candidates(candidates, filters)
    if not filtered_candidates:
        st.info("No candidates match the current filters.")
        return

    grouped_candidates = group_candidates_by_triage(filtered_candidates)
    for triage_label in TRIAGE_LABELS:
        group = grouped_candidates[triage_label]
        st.subheader(f"{TRIAGE_TITLES[triage_label]} ({len(group)})")
        if not group:
            st.caption("No candidates in this triage lane.")
            continue

        for candidate in group:
            render_candidate_card(candidate)


def render_dashboard_metrics(candidates: list[StoredCandidate]) -> None:
    counts = triage_counts(candidates)
    active_count = sum(1 for candidate in candidates if candidate.status == "active")
    columns = st.columns(5)
    columns[0].metric("Total", len(candidates))
    columns[1].metric("Eligible", counts["eligible"])
    columns[2].metric("Not eligible", counts["not_eligible"])
    columns[3].metric("Needs review", counts["needs_review"])
    columns[4].metric("Active", active_count)


def render_dashboard_filters(candidates: list[StoredCandidate]) -> DashboardFilters:
    triage_options = [FILTER_ALL, *TRIAGE_TITLES.values()]
    status_options = [
        FILTER_ALL,
        *sorted({candidate.status for candidate in candidates}),
    ]
    city_options = [
        FILTER_ALL,
        *sorted(
            {
                candidate.profile.city_zone
                for candidate in candidates
                if candidate.profile.city_zone is not None
            }
        ),
    ]

    columns = st.columns([1.1, 1.1, 1.2, 2.2])
    triage = columns[0].selectbox(
        "Triage",
        triage_options,
        key="dashboard_triage_filter",
    )
    status = columns[1].selectbox(
        "Status",
        status_options,
        key="dashboard_status_filter",
    )
    city_zone = columns[2].selectbox(
        "City",
        city_options,
        key="dashboard_city_filter",
    )
    search = columns[3].text_input(
        "Search",
        key="dashboard_search_filter",
        placeholder="Name or candidate ID",
    )

    return DashboardFilters(
        triage=str(triage),
        status=str(status),
        city_zone=str(city_zone),
        search=str(search),
    )


def render_candidate_card(candidate: StoredCandidate) -> None:
    with st.expander(candidate_card_label(candidate), expanded=False):
        header_columns = st.columns([1.1, 1])
        with header_columns[0]:
            st.metric("Triage", TRIAGE_TITLES[candidate_triage(candidate)])
        with header_columns[1]:
            if st.button(
                "Open chat",
                key=f"open_chat_{candidate.id}",
                icon=":material/chat:",
                use_container_width=True,
            ):
                open_candidate_chat(candidate.id)

        st.markdown("**Requirements**")
        requirement_columns = st.columns(3)
        for index, (label, value) in enumerate(candidate_requirement_rows(candidate)):
            with requirement_columns[index % len(requirement_columns)]:
                st.caption(label)
                st.write(value)

        if candidate.profile.missing_fields:
            st.caption("Missing fields")
            st.write(format_list(candidate.profile.missing_fields))
        if candidate.profile.clarification_fields:
            st.caption("Clarification fields")
            st.write(format_list(candidate.profile.clarification_fields))
        if candidate.profile.disqualification_reasons:
            st.caption("Disqualification reasons")
            st.write(format_list(candidate.profile.disqualification_reasons))

        st.caption("HR summary")
        if candidate.hr_summary:
            st.markdown(candidate.hr_summary)
        elif candidate.summary_error:
            st.warning(f"Summary failed: {candidate.summary_error}")
        else:
            st.write("No HR summary yet")
