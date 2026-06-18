"""Streamlit session-state management: agent lifecycle, URL params, and navigation."""

import os
from typing import cast

import streamlit as st

from screening.llm.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV, SCREENING_DB_PATH
from screening.application.session import (
    resume_candidate_session,
    save_current_session,
    start_candidate_session,
)
from screening.domain.models import FINAL_STATUSES
from screening.observability import bind_context
from screening.persistence.storage import StoredCandidate, load_candidate
from screening.ui.constants import AGENT_SESSION_KEY, CANDIDATE_ID_PARAM
from screening.ui.formatting import format_value


def get_or_create_agent() -> ChatAgent:
    """Return the session's chat agent, creating or resuming one as needed.

    Reuses the agent already in session state when present. Otherwise resumes
    the candidate named in the URL query params, falling back to a new session
    if that candidate is missing, and persists the new agent into session state
    and the URL.

    Returns:
        ChatAgent: The active chat agent for this session.
    """
    agent = st.session_state.get(AGENT_SESSION_KEY)
    if agent is not None:
        return cast(ChatAgent, agent)

    candidate_id = query_candidate_id()
    if candidate_id is not None:
        try:
            agent = resume_candidate_session(candidate_id)
        except ValueError:
            bind_context(
                event="candidate_resume_fallback",
                candidate_id=candidate_id,
            ).warning("Candidate session not found; starting a new one")
            st.warning("Candidate session not found. Starting a new one.")
            agent = start_candidate_session()
    else:
        agent = start_candidate_session()

    ensure_agent_started(agent)
    st.session_state[AGENT_SESSION_KEY] = agent
    set_query_candidate_id(agent.candidate_id)
    return agent


def ensure_agent_started(agent: ChatAgent) -> None:
    """Start the agent and persist it if it has no messages yet.

    Args:
        agent (ChatAgent): The agent to start.
    """
    if agent.messages:
        return

    agent.start()
    save_current_session(agent)


def load_current_candidate(agent: ChatAgent) -> StoredCandidate | None:
    """Load the stored candidate record backing the agent.

    Args:
        agent (ChatAgent): The active chat agent.

    Returns:
        StoredCandidate | None: The stored candidate, or None when the agent
            has no candidate identifier.
    """
    if agent.candidate_id is None:
        return None
    return load_candidate(agent.candidate_id, db_path=SCREENING_DB_PATH)


def is_finalized(candidate: StoredCandidate | None) -> bool:
    """Report whether a candidate has reached a final status.

    Args:
        candidate (StoredCandidate | None): The candidate to check.

    Returns:
        bool: True when the candidate exists and its status is final.
    """
    return candidate is not None and candidate.status in FINAL_STATUSES


def query_candidate_id() -> str | None:
    """Read the candidate identifier from the URL query parameters.

    Returns:
        str | None: The candidate id from the query params, or None when absent
            or blank.
    """
    value = st.query_params.get(CANDIDATE_ID_PARAM)
    if value is None:
        return None

    if isinstance(value, list):
        value = value[0] if value else None
    candidate_id = format_value(value)
    return None if candidate_id == "-" else candidate_id


def set_query_candidate_id(candidate_id: str | None) -> None:
    """Write the candidate identifier into the URL query parameters.

    Args:
        candidate_id (str | None): The identifier to set; no-op when None.
    """
    if candidate_id is not None:
        st.query_params[CANDIDATE_ID_PARAM] = candidate_id


def clear_candidate_session() -> None:
    """Clear the active agent and candidate id from session and URL state."""
    st.session_state.pop(AGENT_SESSION_KEY, None)
    try:
        del st.query_params[CANDIDATE_ID_PARAM]
    except KeyError:
        pass


def open_candidate_chat(candidate_id: str) -> None:
    """Switch the chat view to a candidate and rerun the app.

    Clears the current agent, points the URL at the candidate, resumes its
    session when an API key is available, then triggers a rerun.

    Args:
        candidate_id (str): Identifier of the candidate to open.
    """
    st.session_state.pop(AGENT_SESSION_KEY, None)
    set_query_candidate_id(candidate_id)
    if has_api_key():
        st.session_state[AGENT_SESSION_KEY] = resume_candidate_session(candidate_id)
    st.rerun()


def has_api_key() -> bool:
    """Report whether the Anthropic API key environment variable is set.

    Returns:
        bool: True when ``ANTHROPIC_API_KEY`` is present and non-empty.
    """
    has_key = bool(os.getenv(ANTHROPIC_API_KEY_ENV))
    if not has_key:
        bind_context(event="missing_api_key", env_var=ANTHROPIC_API_KEY_ENV).warning(
            "Anthropic API key is missing"
        )
    return has_key
