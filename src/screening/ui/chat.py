"""Streamlit rendering for the candidate chat tab."""

import json
import os
from pathlib import Path

import anthropic
import streamlit as st
from anthropic.types import MessageParam

from screening.llm.agent import ChatAgent
from screening.config import (
    ANTHROPIC_API_KEY_ENV,
    SCREENING_LOG_PATH,
    SCREENING_SHOW_LOG_PANEL_ENV,
)
from screening.application.session import save_current_session
from screening.observability import bind_context, exception_metadata
from screening.ui.constants import CHAT_WINDOW_HEIGHT
from screening.ui.formatting import message_text
from screening.ui.state import get_or_create_agent, has_api_key, is_finalized
from screening.ui.state import load_current_candidate


def render_chat_view() -> ChatAgent | None:
    """Render the candidate chat tab and return its agent.

    Shows an error and renders nothing when no API key is configured.

    Returns:
        ChatAgent | None: The active chat agent, or None when no API key is
            available.
    """
    if not has_api_key():
        st.error(f"{ANTHROPIC_API_KEY_ENV} is missing. The HR dashboard is available.")
        return None

    agent = get_or_create_agent()
    candidate = load_current_candidate(agent)
    render_chat(agent, finalized=is_finalized(candidate))
    return agent


def render_chat(agent: ChatAgent, *, finalized: bool) -> None:
    """Render the chat transcript and handle a new user message.

    Displays existing messages, accepts input when not finalized, streams the
    assistant reply, surfaces memory and extraction warnings, and persists the
    session after each turn.

    Args:
        agent (ChatAgent): The agent driving the conversation.
        finalized (bool): Whether the screening is finalized, which disables
            input.
    """
    messages_container = st.container(
        border=True,
        height=CHAT_WINDOW_HEIGHT,
        key="chat_messages",
        autoscroll=True,
    )
    input_container = st.container()

    with messages_container:
        for message in agent.messages:
            render_message(message)

    with input_container:
        prompt = st.chat_input("Message Lucia", disabled=finalized)
        log_panel_slot = st.empty()

    with messages_container:
        if prompt is None:
            if finalized:
                st.info("Screening finished.")
            render_observability_panel(log_panel_slot, agent.candidate_id)
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
            candidate_message = describe_chat_error(error)
            bind_context(
                event="candidate_chat_error",
                candidate_id=agent.candidate_id,
                **exception_metadata(error),
            ).exception("Candidate-facing chat error")
            st.error(candidate_message)

    render_observability_panel(log_panel_slot, agent.candidate_id)


def describe_chat_error(error: Exception) -> str:
    """Map a chat failure to a candidate-facing message.

    Transient failures (rate limit, overload, timeout, connection) tell the
    candidate to resend, since the SDK's retries have already been exhausted but
    the same message may succeed shortly. Configuration or request errors (4xx)
    say to contact the team, since resending will not help. Non-Anthropic errors
    fall back to a generic message.

    Args:
        error (Exception): The exception raised while streaming the reply.

    Returns:
        str: A message to show the candidate.
    """
    if isinstance(error, anthropic.RateLimitError):
        return "Lucia is busy right now. Please resend in a moment."
    if isinstance(error, anthropic.OverloadedError | anthropic.InternalServerError):
        return "Lucia is temporarily unavailable. Please resend in a moment."
    if isinstance(error, anthropic.APITimeoutError):
        return "That response took too long. Please resend."
    if isinstance(error, anthropic.APIConnectionError):
        return "Could not reach Lucia. Check your connection and resend."
    if isinstance(error, anthropic.APIStatusError):
        return "Lucia can't respond right now. Please contact the recruiting team."
    return f"Lucia could not respond: {error}"


def render_message(message: MessageParam) -> None:
    """Render a single chat message in its role's bubble.

    Args:
        message (MessageParam): The message to display.
    """
    role = "assistant" if message["role"] == "assistant" else "user"
    with st.chat_message(role):
        st.markdown(message_text(message))


def render_observability_panel(slot, candidate_id: str | None) -> None:
    """Render the optional dev-only log panel for the active candidate.

    Args:
        slot: Streamlit placeholder where the panel should be rendered.
        candidate_id (str | None): Candidate id used to filter log records.
    """
    if not show_log_panel_enabled():
        return

    with slot.container():
        with st.expander("Observability", expanded=False):
            rows = candidate_log_rows(candidate_id)
            if not rows:
                st.caption("No log events for this candidate yet.")
                return
            st.dataframe(rows, hide_index=True, use_container_width=True)


def show_log_panel_enabled() -> bool:
    """Report whether the Streamlit log panel should be shown.

    Returns:
        bool: True when ``SCREENING_SHOW_LOG_PANEL`` is truthy.
    """
    return os.getenv(SCREENING_SHOW_LOG_PANEL_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def candidate_log_rows(
    candidate_id: str | None,
    *,
    log_path: str | Path | None = None,
    limit: int = 50,
) -> list[dict[str, object]]:
    """Load compact log rows for a candidate from the JSONL log file.

    Args:
        candidate_id (str | None): Candidate id to filter by.
        log_path (str | Path | None): Optional log file override for tests.
        limit (int): Maximum number of most recent events to return.

    Returns:
        list[dict[str, object]]: Display-ready rows with safe metadata only.
    """
    if candidate_id is None:
        return []

    resolved_log_path = Path(
        log_path or os.getenv("SCREENING_LOG_PATH", SCREENING_LOG_PATH)
    )
    if not resolved_log_path.exists():
        return []

    rows = [
        _candidate_log_row(record)
        for record in _read_log_records(resolved_log_path)
        if _record_extra(record).get("candidate_id") == candidate_id
    ]
    return rows[-limit:]


def _read_log_records(log_path: Path) -> list[dict]:
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        record = payload.get("record")
        if isinstance(record, dict):
            records.append(record)
    return records


def _candidate_log_row(record: dict) -> dict[str, object]:
    extra = _record_extra(record)
    level = record.get("level", {})
    return {
        "time": _record_time(record),
        "level": level.get("name", "-") if isinstance(level, dict) else "-",
        "event": extra.get("event", "-"),
        "operation": extra.get("operation", "-"),
        "model": extra.get("model", "-"),
        "duration_ms": extra.get("duration_ms", "-"),
        "status": _record_status(extra),
        "error": extra.get("exception_type", "-"),
    }


def _record_extra(record: dict) -> dict:
    extra = record.get("extra", {})
    return extra if isinstance(extra, dict) else {}


def _record_time(record: dict) -> str:
    time = record.get("time")
    if isinstance(time, dict):
        return str(time.get("repr", "-"))
    return "-"


def _record_status(extra: dict) -> object:
    if "exception_type" in extra:
        return "error"
    return extra.get("summary_status") or extra.get("status") or "ok"
