import os
from typing import cast

import streamlit as st

from screening.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV, SCREENING_DB_PATH
from screening.session import (
    resume_candidate_session,
    save_current_session,
    start_candidate_session,
)
from screening.models import FINAL_STATUSES
from screening.storage import StoredCandidate, load_candidate
from screening.ui.constants import AGENT_SESSION_KEY, CANDIDATE_ID_PARAM
from screening.ui.formatting import format_value


def get_or_create_agent() -> ChatAgent:
    agent = st.session_state.get(AGENT_SESSION_KEY)
    if agent is not None:
        return cast(ChatAgent, agent)

    candidate_id = query_candidate_id()
    if candidate_id is not None:
        try:
            agent = resume_candidate_session(candidate_id)
        except ValueError:
            st.warning("Candidate session not found. Starting a new one.")
            agent = start_candidate_session()
    else:
        agent = start_candidate_session()

    ensure_agent_started(agent)
    st.session_state[AGENT_SESSION_KEY] = agent
    set_query_candidate_id(agent.candidate_id)
    return agent


def ensure_agent_started(agent: ChatAgent) -> None:
    if agent.messages:
        return

    agent.start()
    save_current_session(agent)


def load_current_candidate(agent: ChatAgent) -> StoredCandidate | None:
    if agent.candidate_id is None:
        return None
    return load_candidate(agent.candidate_id, db_path=SCREENING_DB_PATH)


def is_finalized(candidate: StoredCandidate | None) -> bool:
    return candidate is not None and candidate.status in FINAL_STATUSES


def query_candidate_id() -> str | None:
    value = st.query_params.get(CANDIDATE_ID_PARAM)
    if value is None:
        return None

    if isinstance(value, list):
        value = value[0] if value else None
    candidate_id = format_value(value)
    return None if candidate_id == "-" else candidate_id


def set_query_candidate_id(candidate_id: str | None) -> None:
    if candidate_id is not None:
        st.query_params[CANDIDATE_ID_PARAM] = candidate_id


def clear_candidate_session() -> None:
    st.session_state.pop(AGENT_SESSION_KEY, None)
    try:
        del st.query_params[CANDIDATE_ID_PARAM]
    except KeyError:
        pass


def open_candidate_chat(candidate_id: str) -> None:
    st.session_state.pop(AGENT_SESSION_KEY, None)
    set_query_candidate_id(candidate_id)
    if has_api_key():
        st.session_state[AGENT_SESSION_KEY] = resume_candidate_session(candidate_id)
    st.rerun()


def has_api_key() -> bool:
    return bool(os.getenv(ANTHROPIC_API_KEY_ENV))
