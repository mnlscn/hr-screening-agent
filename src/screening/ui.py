from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from statistics import mean, median
from typing import cast

import altair as alt
import streamlit as st
from anthropic.types import MessageParam
from dotenv import load_dotenv

from screening.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV, SCREENING_DB_PATH
from screening.models import REQUIRED_FIELDS, CandidateProfile, DeliveryExperience
from screening.session import (
    finalize_candidate_session,
    resume_candidate_session,
    save_current_session,
    start_candidate_session,
)
from screening.storage import StoredCandidate, list_candidates, load_candidate


AGENT_SESSION_KEY = "screening_agent"
CANDIDATE_ID_PARAM = "candidate_id"
CHAT_WINDOW_HEIGHT = 600
FILTER_ALL = "All"
TRIAGE_LABELS = ("eligible", "not_eligible", "needs_review")
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

APP_STYLES = """
<style>
    .stApp {
        background: #fbfcfe;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 1.5rem;
    }

    h1 {
        font-size: 2.15rem !important;
        font-weight: 760 !important;
        letter-spacing: 0 !important;
        line-height: 1.1 !important;
        margin-bottom: 0.6rem !important;
    }

    div[data-testid="stTabs"] button {
        font-weight: 650;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #dde5ef !important;
        border-radius: 8px !important;
        box-shadow: 0 12px 30px rgba(31, 41, 55, 0.06);
    }

    div[data-testid="stChatMessage"] {
        border-radius: 8px;
        margin-bottom: 0.35rem;
        padding: 0.55rem 0.75rem;
    }

    div[data-testid="stChatMessage"] p {
        line-height: 1.45;
    }

    div[data-testid="stChatInput"] {
        margin-top: 0.75rem;
    }

    div[data-testid="stSidebar"] {
        background: #f3f6fa;
        border-right: 1px solid #dce4ee;
    }

    div[data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        font-size: 1rem !important;
        letter-spacing: 0 !important;
    }

    div[data-testid="stSidebar"] .stButton > button {
        border-radius: 8px;
        font-weight: 650;
        min-height: 2.45rem;
    }
</style>
"""


@dataclass(frozen=True)
class DashboardFilters:
    triage: str
    status: str
    city_zone: str
    search: str


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


def render_app() -> None:
    load_dotenv()
    st.set_page_config(page_title="Screening", layout="wide")
    _apply_styles()
    st.title("Lucia Screening")

    chat_agent: ChatAgent | None = None
    chat_tab, dashboard_tab, analytics_tab = st.tabs(
        ["Candidate chat", "HR dashboard", "Analytics"]
    )
    with chat_tab:
        chat_agent = _render_chat_view()

    with dashboard_tab:
        _render_dashboard()

    with analytics_tab:
        _render_analytics()

    if chat_agent is None:
        _render_sidebar_without_agent()
    else:
        _render_sidebar(chat_agent)


def _render_chat_view() -> ChatAgent | None:
    if not _has_api_key():
        st.error(f"{ANTHROPIC_API_KEY_ENV} is missing. The HR dashboard is available.")
        return None

    agent = _get_or_create_agent()
    candidate = _load_candidate(agent)
    _render_chat(agent, finalized=_is_finalized(candidate))
    return agent


def _get_or_create_agent() -> ChatAgent:
    agent = st.session_state.get(AGENT_SESSION_KEY)
    if agent is not None:
        return cast(ChatAgent, agent)

    candidate_id = _query_candidate_id()
    if candidate_id is not None:
        try:
            agent = resume_candidate_session(candidate_id)
        except ValueError:
            st.warning("Candidate session not found. Starting a new one.")
            agent = start_candidate_session()
    else:
        agent = start_candidate_session()

    _ensure_agent_started(agent)
    st.session_state[AGENT_SESSION_KEY] = agent
    _set_query_candidate_id(agent.candidate_id)
    return agent


def _ensure_agent_started(agent: ChatAgent) -> None:
    if agent.messages:
        return

    agent.start()
    save_current_session(agent)


def _render_chat(agent: ChatAgent, *, finalized: bool) -> None:
    messages_container = st.container(
        border=True,
        height=CHAT_WINDOW_HEIGHT,
        key="chat_messages",
        autoscroll=True,
    )
    input_container = st.container()

    with messages_container:
        for message in agent.messages:
            _render_message(message)

    with input_container:
        prompt = st.chat_input("Message Lucia", disabled=finalized)

    with messages_container:
        if prompt is None:
            if finalized:
                st.info("Screening finished.")
            return

        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            response = agent.stream(prompt)

            if response.memory_truncated:
                st.warning(
                    "Conversation history was shortened to fit the memory budget."
                )

            if response.input_too_large:
                st.warning("This message is longer than the model input budget.")

            with st.chat_message("assistant"):
                st.write_stream(response.chunks)

            save_current_session(agent)

            if agent.last_extraction_error is not None:
                st.warning(
                    "Candidate extraction failed. The conversation can continue."
                )
        except Exception as error:
            st.error(f"Lucia could not respond: {error}")


def _render_message(message: MessageParam) -> None:
    role = "assistant" if message["role"] == "assistant" else "user"
    with st.chat_message(role):
        st.markdown(_message_text(message))


def _render_sidebar(agent: ChatAgent) -> None:
    candidate = _load_candidate(agent)

    with st.sidebar:
        st.header("Session")

        if st.button(
            "New candidate",
            icon=":material/person_add:",
            use_container_width=True,
        ):
            _clear_candidate_session()
            st.rerun()

        if st.button(
            "Finish screening",
            icon=":material/check_circle:",
            disabled=_is_finalized(candidate),
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Finalizing screening..."):
                finalized = finalize_candidate_session(agent)
            if finalized.summary_status == "completed":
                st.success("Screening finalized.")
            else:
                st.warning("Screening finalized. Summary generation failed.")
            st.rerun()

        st.divider()
        _render_session_metadata(agent, candidate)
        st.divider()
        with st.expander("Profile", expanded=False):
            _render_profile(agent.profile, show_heading=False)
        _render_summary(candidate)


def _render_sidebar_without_agent() -> None:
    with st.sidebar:
        st.header("Session")
        st.warning(f"{ANTHROPIC_API_KEY_ENV} is missing.")
        st.divider()
        st.subheader("Candidate")
        st.text("No active chat session")


def _render_dashboard() -> None:
    candidates = list_candidates(db_path=SCREENING_DB_PATH)
    _render_dashboard_metrics(candidates)

    if not candidates:
        st.info("No candidates in the database yet.")
        return

    filters = _render_dashboard_filters(candidates)
    filtered_candidates = _filter_candidates(candidates, filters)
    if not filtered_candidates:
        st.info("No candidates match the current filters.")
        return

    grouped_candidates = _group_candidates_by_triage(filtered_candidates)
    for triage_label in TRIAGE_LABELS:
        group = grouped_candidates[triage_label]
        st.subheader(f"{TRIAGE_TITLES[triage_label]} ({len(group)})")
        if not group:
            st.caption("No candidates in this triage lane.")
            continue

        for candidate in group:
            _render_candidate_card(candidate)


def _render_analytics() -> None:
    candidates = list_candidates(db_path=SCREENING_DB_PATH)
    if not candidates:
        st.info("No candidates in the database yet.")
        return

    filters = _render_analytics_filters(candidates)
    filtered_candidates = _filter_analytics_candidates(candidates, filters)
    if not filtered_candidates:
        st.info("No candidates match the current analytics filters.")
        return

    _render_analytics_kpis(filtered_candidates)

    st.markdown("### City distribution")
    _render_city_map(filtered_candidates)

    funnel_column, dropoff_column = st.columns(2)
    with funnel_column:
        st.markdown("### Screening funnel")
        _render_ordered_bar_chart(
            _funnel_completion_rows(filtered_candidates),
            category="stage",
            value="completed",
            value_title="Candidates",
            order=_funnel_stage_order(),
            color="#2563EB",
        )
    with dropoff_column:
        st.markdown("### Drop-off stage")
        dropoff_rows = _dropoff_stage_rows(filtered_candidates)
        if dropoff_rows:
            _render_ranked_bar_chart(
                dropoff_rows,
                category="stage",
                value="candidates",
                value_title="Candidates",
                color="#DC2626",
            )
        else:
            st.info("No active or incomplete candidates in this selection.")

    triage_column, availability_column, stale_column = st.columns([1, 1, 1.25])
    with triage_column:
        st.markdown("### Triage mix")
        _render_ranked_bar_chart(
            _analytics_outcome_rows(filtered_candidates),
            category="outcome",
            value="count",
            value_title="Candidates",
            color="#7C3AED",
        )
    with availability_column:
        st.markdown("### Availability mix")
        availability_rows = _availability_rows(filtered_candidates)
        if availability_rows:
            _render_ranked_bar_chart(
                availability_rows,
                category="availability",
                value="count",
                value_title="Candidates",
                color="#0891B2",
            )
        else:
            st.info("No completed or eligible candidates with availability yet.")
    with stale_column:
        st.markdown("### Stale active candidates")
        _render_stale_active_candidates(filtered_candidates)


def _render_analytics_filters(
    candidates: list[StoredCandidate],
) -> AnalyticsFilters:
    started_dates = [
        started_date
        for started_date in (
            _candidate_started_date(candidate) for candidate in candidates
        )
        if started_date is not None
    ]
    default_start_date = min(started_dates) if started_dates else None
    default_end_date = max(started_dates) if started_dates else None
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
    triage_options = [FILTER_ALL, *ANALYTICS_OUTCOME_TITLES.values()]

    columns = st.columns([1, 1, 1.2, 1.3])
    start_date = columns[0].date_input(
        "Started from",
        value=default_start_date,
        key="analytics_start_date",
    )
    end_date = columns[1].date_input(
        "Started to",
        value=default_end_date,
        key="analytics_end_date",
    )
    city_zone = columns[2].selectbox(
        "City",
        city_options,
        key="analytics_city_filter",
    )
    triage = columns[3].selectbox(
        "Triage",
        triage_options,
        key="analytics_triage_filter",
    )

    return AnalyticsFilters(
        start_date=start_date,
        end_date=end_date,
        city_zone=str(city_zone),
        triage=str(triage),
    )


def _render_analytics_kpis(candidates: list[StoredCandidate]) -> None:
    kpis = _analytics_kpis(candidates)
    columns = st.columns(5)
    columns[0].metric("Screenings started", kpis.started)
    columns[1].metric("Completion rate", _format_percent(kpis.completion_rate))
    columns[2].metric("Qualified for HR", kpis.qualified)
    columns[3].metric("Needs review", kpis.needs_review)
    columns[4].metric(
        "Median duration",
        _format_duration(kpis.median_duration_minutes),
    )


def _render_ranked_bar_chart(
    rows: list[dict[str, object]],
    *,
    category: str,
    value: str,
    value_title: str,
    color: str,
) -> None:
    sorted_rows = _ranked_chart_rows(rows, value=value, category=category)
    _render_ordered_bar_chart(
        sorted_rows,
        category=category,
        value=value,
        value_title=value_title,
        order=[str(row[category]) for row in sorted_rows],
        color=color,
    )


def _render_ordered_bar_chart(
    rows: list[dict[str, object]],
    *,
    category: str,
    value: str,
    value_title: str,
    order: list[str],
    color: str,
) -> None:
    if not rows:
        st.info("No data for this chart.")
        return

    data = alt.Data(values=rows)
    base = alt.Chart(data).encode(
        x=alt.X(
            f"{value}:Q",
            title=value_title,
            axis=alt.Axis(grid=True, tickMinStep=1),
        ),
        y=alt.Y(
            f"{category}:N",
            title=None,
            sort=order,
            axis=alt.Axis(labelLimit=180),
        ),
        tooltip=[
            alt.Tooltip(f"{category}:N", title="Metric"),
            alt.Tooltip(f"{value}:Q", title=value_title, format=".0f"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=3, color=color)
    labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        fontSize=12,
        fontWeight="bold",
        color="#334155",
    ).encode(text=alt.Text(f"{value}:Q", format=".0f"))
    chart = (bars + labels).properties(height=max(220, 36 * len(rows)))
    st.altair_chart(chart, use_container_width=True)


def _render_city_map(candidates: list[StoredCandidate]) -> None:
    distribution = _city_distribution_rows(candidates)
    map_rows = [row for row in distribution if row.get("lat") is not None]
    unknown_count = 0
    for row in distribution:
        if row.get("lat") is not None:
            continue
        count = row.get("count")
        if isinstance(count, int):
            unknown_count += count

    if map_rows:
        st.map(
            map_rows,
            latitude="lat",
            longitude="lon",
            color="color",
            size="size",
            height=540,
        )
        st.caption(
            "Bubble size represents candidate volume. Color reflects eligible share "
            "by city."
        )
    else:
        st.info("No mapped candidate cities in this selection.")

    if unknown_count:
        st.caption(f"Unknown or unmapped city: {unknown_count}")

    summary_rows = _city_summary_rows(distribution)
    if summary_rows:
        st.dataframe(summary_rows, hide_index=True, height=180)


def _render_stale_active_candidates(candidates: list[StoredCandidate]) -> None:
    stale_candidates = _stale_active_candidates(candidates)
    if not stale_candidates:
        st.info("No stale active candidates.")
        return

    rows = [
        {
            "Candidate": _candidate_display_name(candidate),
            "City": _format_value(candidate.profile.city_zone),
            "Last update": _format_timestamp(candidate.updated_at),
            "Drop-off": _dropoff_stage(candidate) or "-",
        }
        for candidate in stale_candidates
    ]
    st.dataframe(rows, hide_index=True, height=220)


def _render_dashboard_metrics(candidates: list[StoredCandidate]) -> None:
    counts = _triage_counts(candidates)
    active_count = sum(1 for candidate in candidates if candidate.status == "active")
    columns = st.columns(5)
    columns[0].metric("Total", len(candidates))
    columns[1].metric("Eligible", counts["eligible"])
    columns[2].metric("Not eligible", counts["not_eligible"])
    columns[3].metric("Needs review", counts["needs_review"])
    columns[4].metric("Active", active_count)


def _render_dashboard_filters(candidates: list[StoredCandidate]) -> DashboardFilters:
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


def _render_candidate_card(candidate: StoredCandidate) -> None:
    with st.expander(_candidate_card_label(candidate), expanded=False):
        header_columns = st.columns([1.1, 1])
        with header_columns[0]:
            st.metric("Triage", TRIAGE_TITLES[_candidate_triage(candidate)])
        with header_columns[1]:
            if st.button(
                "Open chat",
                key=f"open_chat_{candidate.id}",
                icon=":material/chat:",
                use_container_width=True,
            ):
                _open_candidate_chat(candidate.id)

        st.markdown("**Requirements**")
        requirement_columns = st.columns(3)
        for index, (label, value) in enumerate(_candidate_requirement_rows(candidate)):
            with requirement_columns[index % len(requirement_columns)]:
                st.caption(label)
                st.write(value)

        if candidate.profile.missing_fields:
            st.caption("Missing fields")
            st.write(_format_list(candidate.profile.missing_fields))
        if candidate.profile.clarification_fields:
            st.caption("Clarification fields")
            st.write(_format_list(candidate.profile.clarification_fields))
        if candidate.profile.disqualification_reasons:
            st.caption("Disqualification reasons")
            st.write(_format_list(candidate.profile.disqualification_reasons))

        st.caption("HR summary")
        if candidate.hr_summary:
            st.markdown(candidate.hr_summary)
        elif candidate.summary_error:
            st.warning(f"Summary failed: {candidate.summary_error}")
        else:
            st.write("No HR summary yet")


def _candidate_card_label(candidate: StoredCandidate) -> str:
    return (
        f"{_candidate_display_name(candidate)} · "
        f"{TRIAGE_TITLES[_candidate_triage(candidate)]} · "
        f"{_short_candidate_id(candidate.id)} · "
        f"Updated {_format_timestamp(candidate.updated_at)}"
    )


def _open_candidate_chat(candidate_id: str) -> None:
    st.session_state.pop(AGENT_SESSION_KEY, None)
    _set_query_candidate_id(candidate_id)
    if _has_api_key():
        st.session_state[AGENT_SESSION_KEY] = resume_candidate_session(candidate_id)
    st.rerun()


def _render_session_metadata(
    agent: ChatAgent,
    candidate: StoredCandidate | None,
) -> None:
    st.subheader("Candidate")
    st.text(f"ID: {_format_value(agent.candidate_id)}")

    if candidate is None:
        st.text("Status: -")
        st.text("Summary: -")
        return

    st.text(f"Status: {candidate.status}")
    st.text(f"Summary: {_format_value(candidate.summary_status)}")
    st.text(f"Started: {_format_timestamp(candidate.started_at)}")
    st.text(f"Updated: {_format_timestamp(candidate.updated_at)}")


def _render_profile(profile: CandidateProfile, *, show_heading: bool = True) -> None:
    if show_heading:
        st.subheader("Profile")

    for label, value in _profile_rows(profile):
        st.text(f"{label}: {value}")

    st.caption("Missing fields")
    st.text(_format_list(profile.missing_fields))
    st.caption("Clarification fields")
    st.text(_format_list(profile.clarification_fields))
    st.caption("Disqualification reasons")
    st.text(_format_list(profile.disqualification_reasons))


def _render_summary(candidate: StoredCandidate | None) -> None:
    if candidate is None:
        return
    if candidate.bot_label is None and candidate.hr_summary is None:
        return

    st.divider()
    st.subheader("HR")
    st.text(f"Bot label: {_format_value(candidate.bot_label)}")
    if candidate.hr_summary:
        st.caption("Summary")
        st.markdown(candidate.hr_summary)
    if candidate.summary_error:
        st.caption("Summary error")
        st.code(candidate.summary_error)


def _profile_rows(profile: CandidateProfile) -> list[tuple[str, str]]:
    return [
        ("Full name", _format_value(profile.full_name)),
        ("Drivers license", _format_value(profile.drivers_license)),
        ("Raw city or zone", _format_value(profile.raw_city_zone)),
        ("City or zone", _format_value(profile.city_zone)),
        ("City status", _format_value(profile.city_zone_status)),
        ("Language", _format_value(profile.conversation_language)),
        ("Availability", _format_value(profile.availability)),
        ("Schedule", _format_value(profile.preferred_schedule)),
        ("Experience", _format_experience(profile.prior_delivery_experience)),
        ("Start date", _format_value(profile.start_date)),
    ]


def _candidate_requirement_rows(candidate: StoredCandidate) -> list[tuple[str, str]]:
    profile = candidate.profile
    return [
        ("Status", _format_value(candidate.status)),
        ("License", _format_value(profile.drivers_license)),
        ("City", _format_value(profile.city_zone)),
        ("Availability", _format_value(profile.availability)),
        ("Schedule", _format_value(profile.preferred_schedule)),
        ("Experience", _format_experience(profile.prior_delivery_experience)),
        ("Start date", _format_value(profile.start_date)),
        ("Summary", _format_value(candidate.summary_status)),
        ("Label", TRIAGE_TITLES[_candidate_triage(candidate)]),
    ]


def _ranked_chart_rows(
    rows: list[dict[str, object]],
    *,
    value: str,
    category: str,
) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (-_numeric_row_value(row, value), str(row[category])),
    )


def _numeric_row_value(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0


def _filter_analytics_candidates(
    candidates: list[StoredCandidate],
    filters: AnalyticsFilters,
) -> list[StoredCandidate]:
    return [
        candidate
        for candidate in candidates
        if _candidate_matches_analytics_filters(candidate, filters)
    ]


def _candidate_matches_analytics_filters(
    candidate: StoredCandidate,
    filters: AnalyticsFilters,
) -> bool:
    started_date = _candidate_started_date(candidate)
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
    outcome = _analytics_outcome(candidate)
    if (
        filters.triage != FILTER_ALL
        and filters.triage != ANALYTICS_OUTCOME_TITLES[outcome]
    ):
        return False
    return True


def _analytics_kpis(candidates: list[StoredCandidate]) -> AnalyticsKpis:
    started = len(candidates)
    completed = sum(1 for candidate in candidates if candidate.status == "completed")
    duration_stats = _duration_stats(candidates)
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
            if _candidate_triage(candidate) == "needs_review"
        ),
        average_duration_minutes=duration_stats.average_minutes,
        median_duration_minutes=duration_stats.median_minutes,
    )


def _duration_stats(candidates: list[StoredCandidate]) -> DurationStats:
    durations = [
        duration
        for duration in (
            _candidate_duration_minutes(candidate) for candidate in candidates
        )
        if duration is not None
    ]
    if not durations:
        return DurationStats(average_minutes=None, median_minutes=None)
    return DurationStats(
        average_minutes=mean(durations),
        median_minutes=median(durations),
    )


def _candidate_duration_minutes(candidate: StoredCandidate) -> float | None:
    if candidate.completed_at is None:
        return None
    started_at = _parse_timestamp(candidate.started_at)
    completed_at = _parse_timestamp(candidate.completed_at)
    if started_at is None or completed_at is None:
        return None
    duration = completed_at - started_at
    if duration.total_seconds() < 0:
        return None
    return duration.total_seconds() / 60


def _funnel_completion_rows(
    candidates: list[StoredCandidate],
) -> list[dict[str, object]]:
    return [
        {
            "stage": FIELD_TITLES[field],
            "completed": sum(
                1
                for candidate in candidates
                if _profile_field_is_complete(candidate, field)
            ),
        }
        for field in REQUIRED_FIELDS
    ]


def _funnel_stage_order() -> list[str]:
    return [FIELD_TITLES[field] for field in REQUIRED_FIELDS]


def _profile_field_is_complete(candidate: StoredCandidate, field: str) -> bool:
    profile = candidate.profile
    return (
        field not in profile.missing_fields
        and field not in profile.clarification_fields
    )


def _dropoff_stage_rows(candidates: list[StoredCandidate]) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        stage = _dropoff_stage(candidate)
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


def _dropoff_stage(candidate: StoredCandidate) -> str | None:
    if candidate.status != "active" and candidate.profile.is_complete:
        return None
    if candidate.profile.clarification_fields:
        return candidate.profile.clarification_fields[0]
    if candidate.profile.missing_fields:
        return candidate.profile.missing_fields[0]
    return None


def _city_distribution_rows(
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
                "size": _city_bubble_size(count),
                "color": _eligible_share_color(eligible_share),
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


def _city_summary_rows(
    distribution_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "City": str(row["city"]),
            "Candidates": row["count"],
            "Eligible": row["eligible_count"],
            "Eligible share": _format_percent(
                _numeric_row_value(row, "eligible_share")
            ),
        }
        for row in sorted(
            distribution_rows,
            key=lambda row: (-_numeric_row_value(row, "count"), str(row["city"])),
        )
    ]


def _city_bubble_size(count: int) -> int:
    return max(MAP_BUBBLE_SIZE_SCALE, count * MAP_BUBBLE_SIZE_SCALE)


def _eligible_share_color(eligible_share: float) -> str:
    if eligible_share >= 0.5:
        return "#16A34A"
    if eligible_share > 0:
        return "#F59E0B"
    return "#EF4444"


def _analytics_outcome_rows(
    candidates: list[StoredCandidate],
) -> list[dict[str, object]]:
    counts = {outcome: 0 for outcome in ANALYTICS_OUTCOME_TITLES}
    for candidate in candidates:
        counts[_analytics_outcome(candidate)] += 1
    return [
        {"outcome": title, "count": counts[outcome]}
        for outcome, title in ANALYTICS_OUTCOME_TITLES.items()
        if counts[outcome] > 0
    ]


def _analytics_outcome(candidate: StoredCandidate) -> str:
    if candidate.status == "active" and candidate.bot_label is None:
        return "active_without_summary"
    if candidate.bot_label in {"eligible", "not_eligible"}:
        return candidate.bot_label
    return "needs_review"


def _availability_rows(candidates: list[StoredCandidate]) -> list[dict[str, object]]:
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


def _stale_active_candidates(
    candidates: list[StoredCandidate],
    *,
    now: datetime | None = None,
) -> list[StoredCandidate]:
    now = now or datetime.now(UTC)
    stale_candidates = []
    for candidate in candidates:
        if candidate.status != "active":
            continue
        updated_at = _parse_timestamp(candidate.updated_at)
        if updated_at is None:
            continue
        if now - updated_at > STALE_ACTIVE_AFTER:
            stale_candidates.append(candidate)
    return stale_candidates


def _filter_candidates(
    candidates: list[StoredCandidate],
    filters: DashboardFilters,
) -> list[StoredCandidate]:
    normalized_search = filters.search.strip().lower()
    return [
        candidate
        for candidate in candidates
        if _candidate_matches_filters(candidate, filters, normalized_search)
    ]


def _candidate_matches_filters(
    candidate: StoredCandidate,
    filters: DashboardFilters,
    normalized_search: str,
) -> bool:
    triage = _candidate_triage(candidate)
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
            _candidate_display_name(candidate),
            _format_value(candidate.profile.city_zone),
            _format_value(candidate.hr_summary),
        )
    ).lower()
    return normalized_search in searchable_text


def _group_candidates_by_triage(
    candidates: list[StoredCandidate],
) -> dict[str, list[StoredCandidate]]:
    grouped = {triage_label: [] for triage_label in TRIAGE_LABELS}
    for candidate in candidates:
        grouped[_candidate_triage(candidate)].append(candidate)
    return grouped


def _triage_counts(candidates: list[StoredCandidate]) -> dict[str, int]:
    counts = {triage_label: 0 for triage_label in TRIAGE_LABELS}
    for candidate in candidates:
        counts[_candidate_triage(candidate)] += 1
    return counts


def _candidate_triage(candidate: StoredCandidate) -> str:
    if candidate.bot_label in TRIAGE_LABELS:
        return candidate.bot_label
    return "needs_review"


def _candidate_display_name(candidate: StoredCandidate) -> str:
    return _format_value(candidate.profile.full_name).replace("-", "") or (
        f"Candidate {_short_candidate_id(candidate.id)}"
    )


def _short_candidate_id(candidate_id: str) -> str:
    if len(candidate_id) <= 12:
        return candidate_id
    return f"{candidate_id[:8]}...{candidate_id[-4:]}"


def _format_experience(experience: DeliveryExperience | None) -> str:
    if experience is None:
        return "-"

    values = []
    if experience.years is not None:
        values.append(f"{experience.years:g} years")
    if experience.platform is not None:
        values.append(experience.platform)

    return ", ".join(values) if values else "-"


def _format_value(value: object) -> str:
    if value is None:
        return "-"
    value_text = str(value).strip()
    return value_text if value_text else "-"


def _format_list(values: Iterable[str]) -> str:
    formatted_values = [value for value in values if value]
    return ", ".join(formatted_values) if formatted_values else "-"


def _format_timestamp(value: str | None) -> str:
    if value is None:
        return "-"
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return _format_value(value)
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _candidate_started_date(candidate: StoredCandidate) -> date | None:
    started_at = _parse_timestamp(candidate.started_at)
    return started_at.date() if started_at is not None else None


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _format_percent(value: float) -> str:
    return f"{value:.0%}"


def _format_duration(minutes: float | None) -> str:
    if minutes is None:
        return "-"
    if minutes < 60:
        return f"{minutes:.0f} min"
    hours = minutes / 60
    return f"{hours:.1f} h"


def _message_text(message: MessageParam) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return str(content)


def _load_candidate(agent: ChatAgent) -> StoredCandidate | None:
    if agent.candidate_id is None:
        return None
    return load_candidate(agent.candidate_id, db_path=SCREENING_DB_PATH)


def _is_finalized(candidate: StoredCandidate | None) -> bool:
    return candidate is not None and candidate.summary_status is not None


def _query_candidate_id() -> str | None:
    value = st.query_params.get(CANDIDATE_ID_PARAM)
    if value is None:
        return None

    if isinstance(value, list):
        value = value[0] if value else None
    candidate_id = _format_value(value)
    return None if candidate_id == "-" else candidate_id


def _set_query_candidate_id(candidate_id: str | None) -> None:
    if candidate_id is not None:
        st.query_params[CANDIDATE_ID_PARAM] = candidate_id


def _clear_candidate_session() -> None:
    st.session_state.pop(AGENT_SESSION_KEY, None)
    try:
        del st.query_params[CANDIDATE_ID_PARAM]
    except KeyError:
        pass


def _apply_styles() -> None:
    st.markdown(APP_STYLES, unsafe_allow_html=True)


def _has_api_key() -> bool:
    return bool(os.getenv(ANTHROPIC_API_KEY_ENV))


if __name__ == "__main__":
    render_app()
