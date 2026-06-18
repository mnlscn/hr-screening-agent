from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import streamlit as st
from anthropic.types import MessageParam
from dotenv import load_dotenv

from screening.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV, SCREENING_DB_PATH
from screening.models import CandidateProfile, DeliveryExperience
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


def render_app() -> None:
    load_dotenv()
    st.set_page_config(page_title="Screening", layout="wide")
    _apply_styles()
    st.title("Lucia Screening")

    chat_agent: ChatAgent | None = None
    chat_tab, dashboard_tab = st.tabs(["Candidate chat", "HR dashboard"])
    with chat_tab:
        chat_agent = _render_chat_view()

    with dashboard_tab:
        _render_dashboard()

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
        _render_profile(agent.profile)
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
    with st.container(border=True):
        header_columns = st.columns([3, 1.1, 1])
        with header_columns[0]:
            st.markdown(f"#### {_candidate_display_name(candidate)}")
            st.caption(
                f"{_short_candidate_id(candidate.id)} · "
                f"Updated {_format_timestamp(candidate.updated_at)}"
            )
        with header_columns[1]:
            st.metric("Triage", TRIAGE_TITLES[_candidate_triage(candidate)])
        with header_columns[2]:
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
            st.write(candidate.hr_summary)
        elif candidate.summary_error:
            st.warning(f"Summary failed: {candidate.summary_error}")
        else:
            st.write("No HR summary yet")


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


def _render_profile(profile: CandidateProfile) -> None:
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
        st.write(candidate.hr_summary)
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
