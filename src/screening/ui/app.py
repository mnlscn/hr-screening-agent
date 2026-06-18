"""Streamlit app entry point: assembles the chat, dashboard, and analytics tabs."""

import streamlit as st
from dotenv import load_dotenv

from screening.llm.agent import ChatAgent
from screening.observability import configure_logging
from screening.ui.analytics import render_analytics
from screening.ui.chat import render_chat_view
from screening.ui.dashboard import render_dashboard
from screening.ui.sidebar import render_sidebar, render_sidebar_without_agent
from screening.ui.styles import APP_STYLES


def render_app() -> None:
    """Render the full Streamlit app: chat, dashboard, and analytics tabs.

    Loads environment variables, configures the page and styles, builds the
    three tabs, and renders the sidebar based on whether a chat agent is
    active.
    """
    load_dotenv()
    configure_logging()
    st.set_page_config(page_title="Screening", layout="wide")
    apply_styles()
    st.title("Lucia Screening")

    chat_agent: ChatAgent | None = None
    chat_tab, dashboard_tab, analytics_tab = st.tabs(
        ["Candidate chat", "HR dashboard", "Analytics"]
    )
    with chat_tab:
        chat_agent = render_chat_view()

    with dashboard_tab:
        render_dashboard()

    with analytics_tab:
        render_analytics()

    if chat_agent is None:
        render_sidebar_without_agent()
    else:
        render_sidebar(chat_agent)


def apply_styles() -> None:
    """Inject the app's custom CSS into the Streamlit page."""
    st.markdown(APP_STYLES, unsafe_allow_html=True)


if __name__ == "__main__":
    render_app()
