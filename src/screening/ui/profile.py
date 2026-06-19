"""Shared rendering for a candidate profile block."""

import streamlit as st

from screening.domain.models import CandidateProfile
from screening.ui.formatting import format_list, profile_rows


def render_profile(profile: CandidateProfile, *, show_heading: bool = True) -> None:
    """Render a candidate profile with its derived field lists.

    Args:
        profile (CandidateProfile): The profile to render.
        show_heading (bool): Whether to render a "Profile" subheading.
    """
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
