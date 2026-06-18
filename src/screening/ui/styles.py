"""Custom CSS injected into the Streamlit app for layout and visual polish."""

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
