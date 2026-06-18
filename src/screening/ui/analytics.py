import altair as alt
import streamlit as st

from screening.config import SCREENING_DB_PATH
from screening.storage import StoredCandidate, list_candidates
from screening.ui.analytics_logic import (
    AnalyticsFilters,
    analytics_kpis,
    analytics_outcome_rows,
    availability_rows,
    city_distribution_rows,
    city_summary_rows,
    dropoff_stage,
    dropoff_stage_rows,
    filter_analytics_candidates,
    funnel_completion_rows,
    funnel_stage_order,
    ranked_chart_rows,
    stale_active_candidates,
)
from screening.ui.constants import ANALYTICS_OUTCOME_TITLES, FILTER_ALL
from screening.ui.dashboard_logic import candidate_display_name
from screening.ui.formatting import (
    candidate_started_date,
    format_duration,
    format_timestamp,
    format_value,
)


def render_analytics() -> None:
    candidates = list_candidates(db_path=SCREENING_DB_PATH)
    if not candidates:
        st.info("No candidates in the database yet.")
        return

    filters = render_analytics_filters(candidates)
    filtered_candidates = filter_analytics_candidates(candidates, filters)
    if not filtered_candidates:
        st.info("No candidates match the current analytics filters.")
        return

    render_analytics_kpis(filtered_candidates)

    st.markdown("### City distribution")
    render_city_map(filtered_candidates)

    funnel_column, dropoff_column = st.columns(2)
    with funnel_column:
        st.markdown("### Screening funnel")
        render_ordered_bar_chart(
            funnel_completion_rows(filtered_candidates),
            category="stage",
            value="completed",
            value_title="Candidates",
            order=funnel_stage_order(),
            color="#2563EB",
        )
    with dropoff_column:
        st.markdown("### Drop-off stage")
        dropoff_rows = dropoff_stage_rows(filtered_candidates)
        if dropoff_rows:
            render_ranked_bar_chart(
                dropoff_rows,
                category="stage",
                value="candidates",
                value_title="Candidates",
                color="#DC2626",
            )
        else:
            st.info("No active or incomplete candidates in this selection.")

    triage_column, availability_column, stale_column = st.columns([1, 1, 1.25])
    with triage_column:
        st.markdown("### Triage mix")
        render_ranked_bar_chart(
            analytics_outcome_rows(filtered_candidates),
            category="outcome",
            value="count",
            value_title="Candidates",
            color="#7C3AED",
        )
    with availability_column:
        st.markdown("### Availability mix")
        rows = availability_rows(filtered_candidates)
        if rows:
            render_ranked_bar_chart(
                rows,
                category="availability",
                value="count",
                value_title="Candidates",
                color="#0891B2",
            )
        else:
            st.info("No completed or eligible candidates with availability yet.")
    with stale_column:
        st.markdown("### Stale active candidates")
        render_stale_active_candidates(filtered_candidates)


def render_analytics_filters(
    candidates: list[StoredCandidate],
) -> AnalyticsFilters:
    started_dates = [
        started_date
        for started_date in (
            candidate_started_date(candidate.started_at) for candidate in candidates
        )
        if started_date is not None
    ]
    default_start_date = min(started_dates) if started_dates else None
    default_end_date = max(started_dates) if started_dates else None
    city_options = [
        FILTER_ALL,
        *sorted(
            {
                candidate.profile.city_zone
                for candidate in candidates
                if candidate.profile.city_zone is not None
            }
        ),
    ]
    triage_options = [FILTER_ALL, *ANALYTICS_OUTCOME_TITLES.values()]

    columns = st.columns([1, 1, 1.2, 1.3])
    start_date = columns[0].date_input(
        "Started from",
        value=default_start_date,
        key="analytics_start_date",
    )
    end_date = columns[1].date_input(
        "Started to",
        value=default_end_date,
        key="analytics_end_date",
    )
    city_zone = columns[2].selectbox(
        "City",
        city_options,
        key="analytics_city_filter",
    )
    triage = columns[3].selectbox(
        "Triage",
        triage_options,
        key="analytics_triage_filter",
    )

    return AnalyticsFilters(
        start_date=start_date,
        end_date=end_date,
        city_zone=str(city_zone),
        triage=str(triage),
    )


def render_analytics_kpis(candidates: list[StoredCandidate]) -> None:
    kpis = analytics_kpis(candidates)
    columns = st.columns(5)
    columns[0].metric("Screenings started", kpis.started)
    columns[1].metric("Completion rate", f"{kpis.completion_rate:.0%}")
    columns[2].metric("Qualified for HR", kpis.qualified)
    columns[3].metric("Needs review", kpis.needs_review)
    columns[4].metric(
        "Median duration",
        format_duration(kpis.median_duration_minutes),
    )


def render_ranked_bar_chart(
    rows: list[dict[str, object]],
    *,
    category: str,
    value: str,
    value_title: str,
    color: str,
) -> None:
    sorted_rows = ranked_chart_rows(rows, value=value, category=category)
    render_ordered_bar_chart(
        sorted_rows,
        category=category,
        value=value,
        value_title=value_title,
        order=[str(row[category]) for row in sorted_rows],
        color=color,
    )


def render_ordered_bar_chart(
    rows: list[dict[str, object]],
    *,
    category: str,
    value: str,
    value_title: str,
    order: list[str],
    color: str,
) -> None:
    if not rows:
        st.info("No data for this chart.")
        return

    data = alt.Data(values=rows)
    base = alt.Chart(data).encode(
        x=alt.X(
            f"{value}:Q",
            title=value_title,
            axis=alt.Axis(grid=True, tickMinStep=1),
        ),
        y=alt.Y(
            f"{category}:N",
            title=None,
            sort=order,
            axis=alt.Axis(labelLimit=180),
        ),
        tooltip=[
            alt.Tooltip(f"{category}:N", title="Metric"),
            alt.Tooltip(f"{value}:Q", title=value_title, format=".0f"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=3, color=color)
    labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        fontSize=12,
        fontWeight="bold",
        color="#334155",
    ).encode(text=alt.Text(f"{value}:Q", format=".0f"))
    chart = (bars + labels).properties(height=max(220, 36 * len(rows)))
    st.altair_chart(chart, use_container_width=True)


def render_city_map(candidates: list[StoredCandidate]) -> None:
    distribution = city_distribution_rows(candidates)
    map_rows = [row for row in distribution if row.get("lat") is not None]
    unknown_count = 0
    for row in distribution:
        if row.get("lat") is not None:
            continue
        count = row.get("count")
        if isinstance(count, int):
            unknown_count += count

    if map_rows:
        st.map(
            map_rows,
            latitude="lat",
            longitude="lon",
            color="color",
            size="size",
            height=540,
        )
        st.caption(
            "Bubble size represents candidate volume. Color reflects eligible share "
            "by city."
        )
    else:
        st.info("No mapped candidate cities in this selection.")

    if unknown_count:
        st.caption(f"Unknown or unmapped city: {unknown_count}")

    summary_rows = city_summary_rows(distribution)
    if summary_rows:
        st.dataframe(summary_rows, hide_index=True, height=180)


def render_stale_active_candidates(candidates: list[StoredCandidate]) -> None:
    stale_candidates = stale_active_candidates(candidates)
    if not stale_candidates:
        st.info("No stale active candidates.")
        return

    rows = [
        {
            "Candidate": candidate_display_name(candidate),
            "City": format_value(candidate.profile.city_zone),
            "Last update": format_timestamp(candidate.updated_at),
            "Drop-off": dropoff_stage(candidate) or "-",
        }
        for candidate in stale_candidates
    ]
    st.dataframe(rows, hide_index=True, height=220)
