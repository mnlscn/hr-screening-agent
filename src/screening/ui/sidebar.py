import streamlit as st

from screening.llm.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV
from screening.domain.models import CandidateProfile
from screening.application.session import finalize_candidate_session
from screening.persistence.storage import StoredCandidate
from screening.ui.formatting import format_list, format_value, format_timestamp
from screening.ui.formatting import profile_rows
from screening.ui.state import clear_candidate_session, is_finalized
from screening.ui.state import load_current_candidate


def render_sidebar(agent: ChatAgent) -> None:
    candidate = load_current_candidate(agent)

    with st.sidebar:
        st.header("Session")

        if st.button(
            "New candidate",
            icon=":material/person_add:",
            use_container_width=True,
        ):
            clear_candidate_session()
            st.rerun()

        if st.button(
            "Finish screening",
            icon=":material/check_circle:",
            disabled=is_finalized(candidate),
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
        render_session_metadata(agent, candidate)
        st.divider()
        with st.expander("Profile", expanded=False):
            render_profile(agent.profile, show_heading=False)
        render_summary(candidate)


def render_sidebar_without_agent() -> None:
    with st.sidebar:
        st.header("Session")
        st.warning(f"{ANTHROPIC_API_KEY_ENV} is missing.")
        st.divider()
        st.subheader("Candidate")
        st.text("No active chat session")


def render_session_metadata(
    agent: ChatAgent,
    candidate: StoredCandidate | None,
) -> None:
    st.subheader("Candidate")
    st.text(f"ID: {format_value(agent.candidate_id)}")

    if candidate is None:
        st.text("Status: -")
        st.text("Summary: -")
        return

    st.text(f"Status: {candidate.status}")
    st.text(f"Summary: {format_value(candidate.summary_status)}")
    st.text(f"Started: {format_timestamp(candidate.started_at)}")
    st.text(f"Updated: {format_timestamp(candidate.updated_at)}")


def render_profile(profile: CandidateProfile, *, show_heading: bool = True) -> None:
    if show_heading:
        st.subheader("Profile")

    for label, value in profile_rows(profile):
        st.text(f"{label}: {value}")

    st.caption("Missing fields")
    st.text(format_list(profile.missing_fields))
    st.caption("Clarification fields")
    st.text(format_list(profile.clarification_fields))
    st.caption("Disqualification reasons")
    st.text(format_list(profile.disqualification_reasons))


def render_summary(candidate: StoredCandidate | None) -> None:
    if candidate is None:
        return
    if candidate.bot_label is None and candidate.hr_summary is None:
        return

    st.divider()
    st.subheader("HR")
    st.text(f"Bot label: {format_value(candidate.bot_label)}")
    if candidate.hr_summary:
        st.caption("Summary")
        st.markdown(candidate.hr_summary)
    if candidate.summary_error:
        st.caption("Summary error")
        st.code(candidate.summary_error)
