import streamlit as st
from dotenv import load_dotenv

from screening.llm.agent import ChatAgent
from screening.ui.analytics import render_analytics
from screening.ui.chat import render_chat_view
from screening.ui.dashboard import render_dashboard
from screening.ui.sidebar import render_sidebar, render_sidebar_without_agent
from screening.ui.styles import APP_STYLES


def render_app() -> None:
    load_dotenv()
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
    st.markdown(APP_STYLES, unsafe_allow_html=True)


if __name__ == "__main__":
    render_app()
