import streamlit as st
from anthropic.types import MessageParam

from screening.llm.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV
from screening.application.session import save_current_session
from screening.ui.constants import CHAT_WINDOW_HEIGHT
from screening.ui.formatting import message_text
from screening.ui.state import get_or_create_agent, has_api_key, is_finalized
from screening.ui.state import load_current_candidate


def render_chat_view() -> ChatAgent | None:
    if not has_api_key():
        st.error(f"{ANTHROPIC_API_KEY_ENV} is missing. The HR dashboard is available.")
        return None

    agent = get_or_create_agent()
    candidate = load_current_candidate(agent)
    render_chat(agent, finalized=is_finalized(candidate))
    return agent


def render_chat(agent: ChatAgent, *, finalized: bool) -> None:
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


def render_message(message: MessageParam) -> None:
    role = "assistant" if message["role"] == "assistant" else "user"
    with st.chat_message(role):
        st.markdown(message_text(message))
