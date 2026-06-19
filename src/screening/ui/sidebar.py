"""Streamlit sidebar: session controls and a WhatsApp-style candidate list."""

import streamlit as st

from screening.llm.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV, SCREENING_DB_PATH
from screening.application.session import finalize_candidate_session
from screening.persistence.storage import list_candidates
from screening.ui.dashboard_logic import candidate_display_name
from screening.ui.state import clear_candidate_session, is_finalized
from screening.ui.state import load_current_candidate, open_candidate_chat


def render_sidebar(agent: ChatAgent) -> None:
    """Render the session sidebar for an active chat agent.

    Provides controls to start a new candidate and finish the screening, and
    shows a clickable list of candidates that switches the chat on click.

    Args:
        agent (ChatAgent): The active chat agent.
    """
    candidate = load_current_candidate(agent)

    with st.sidebar:
        st.header("Chats")

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
        render_candidate_list(agent)


def render_sidebar_without_agent() -> None:
    """Render the sidebar fallback shown when no chat agent is active."""
    with st.sidebar:
        st.header("Chats")
        st.warning(f"{ANTHROPIC_API_KEY_ENV} is missing.")
        st.divider()
        st.text("No active chat session")


def render_candidate_list(agent: ChatAgent) -> None:
    """Render a clickable list of candidates, highlighting the active one.

    Each row shows an avatar icon and the candidate's name; clicking it resumes
    and opens that candidate's chat. Candidates are listed most-recent first.

    Args:
        agent (ChatAgent): The active chat agent, whose candidate is highlighted.
    """
    candidates = list_candidates(db_path=SCREENING_DB_PATH)
    active_id = agent.candidate_id

    with st.container(key="candidate_list"):
        if not candidates:
            st.caption("No candidates yet.")
            return

        for candidate in candidates:
            if st.button(
                candidate_display_name(candidate),
                key=f"sidebar_candidate_{candidate.id}",
                icon=":material/account_circle:",
                use_container_width=True,
                type="primary" if candidate.id == active_id else "secondary",
            ):
                open_candidate_chat(candidate.id)
