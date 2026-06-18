from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from anthropic.types import MessageParam
from streamlit.testing.v1 import AppTest

import screening.ui.analytics as ui_analytics
import screening.ui.analytics_logic as analytics_logic
import screening.ui.app as ui_app
import screening.ui.chat as ui_chat
import screening.ui.constants as ui_constants
import screening.ui.dashboard as ui_dashboard
import screening.ui.dashboard_logic as dashboard_logic
import screening.ui.formatting as ui_formatting
import screening.ui.sidebar as ui_sidebar
import screening.ui.state as ui_state
from screening.agent import AgentStream
from screening.config import ANTHROPIC_API_KEY_ENV
from screening.models import CandidateProfile, DeliveryExperience, DriverLicense
from screening.session import FinalizedCandidateSession
from screening.storage import StoredCandidate


class FakeUiAgent:
    def __init__(self, candidate_id: str = "candidate-1"):
        self.candidate_id = candidate_id
        self.profile = CandidateProfile(full_name="Maria Garcia")
        self.messages = [cast(MessageParam, {"role": "assistant", "content": "Hola"})]
        self.last_extraction_error = None

    def start(self):
        return AgentStream(chunks=iter(()))

    def stream(self, user_input: str):
        self.messages.append(
            cast(MessageParam, {"role": "user", "content": user_input})
        )
        self.messages.append(
            cast(MessageParam, {"role": "assistant", "content": "Respuesta"})
        )
        self.profile = self.profile.merge(CandidateProfile(availability="Full-time"))
        return AgentStream(chunks=iter(["Respuesta"]))


def test_profile_rows_format_extracted_fields():
    profile = CandidateProfile(
        full_name="Maria Garcia",
        drivers_license="Yes",
        raw_city_zone="Madrid centro",
        city_zone="Madrid",
        city_zone_status="Matched",
        conversation_language="Spanish",
        availability="Full-time",
        preferred_schedule="Morning",
        prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
        start_date="next Monday",
    )

    rows = dict(ui_formatting.profile_rows(profile))

    assert rows["Full name"] == "Maria Garcia"
    assert rows["Experience"] == "2 years, Glovo"
    assert rows["Availability"] == "Full-time"


def test_render_app_uses_injected_agent_without_anthropic(monkeypatch):
    agent = FakeUiAgent()
    candidate = _stored_candidate(agent)

    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "test-key")
    monkeypatch.setattr(ui_state, "start_candidate_session", lambda: agent)
    monkeypatch.setattr(
        ui_state,
        "load_candidate",
        lambda candidate_id, *, db_path: candidate,
    )
    monkeypatch.setattr(ui_dashboard, "list_candidates", lambda *, db_path: [])
    monkeypatch.setattr(ui_analytics, "list_candidates", lambda *, db_path: [])

    app = AppTest.from_function(_run_ui_app)
    app.run()

    assert not app.exception
    assert app.title[0].value == "Lucia Screening"
    assert any("Hola" in markdown.value for markdown in app.markdown)


def test_chat_submit_streams_and_saves_with_injected_agent(monkeypatch):
    agent = FakeUiAgent()
    candidate = _stored_candidate(agent)
    saved_messages: list[list[MessageParam]] = []

    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "test-key")
    monkeypatch.setattr(ui_state, "start_candidate_session", lambda: agent)
    monkeypatch.setattr(
        ui_state,
        "load_candidate",
        lambda candidate_id, *, db_path: candidate,
    )
    monkeypatch.setattr(ui_dashboard, "list_candidates", lambda *, db_path: [])
    monkeypatch.setattr(ui_analytics, "list_candidates", lambda *, db_path: [])
    monkeypatch.setattr(
        ui_chat,
        "save_current_session",
        lambda saved_agent: saved_messages.append(list(saved_agent.messages)),
    )

    app = AppTest.from_function(_run_ui_app)
    app.run()
    app.chat_input[0].set_value("Tengo carnet").run(timeout=10)

    assert not app.exception
    assert saved_messages[-1] == [
        cast(MessageParam, {"role": "assistant", "content": "Hola"}),
        cast(MessageParam, {"role": "user", "content": "Tengo carnet"}),
        cast(MessageParam, {"role": "assistant", "content": "Respuesta"}),
    ]


def test_finish_button_finalizes_with_injected_agent(monkeypatch):
    agent = FakeUiAgent()
    candidate = _stored_candidate(agent)
    finalized_candidate_ids: list[str] = []

    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "test-key")
    monkeypatch.setattr(ui_state, "start_candidate_session", lambda: agent)
    monkeypatch.setattr(
        ui_state,
        "load_candidate",
        lambda candidate_id, *, db_path: candidate,
    )
    monkeypatch.setattr(ui_dashboard, "list_candidates", lambda *, db_path: [])
    monkeypatch.setattr(ui_analytics, "list_candidates", lambda *, db_path: [])

    def fake_finalize(saved_agent):
        finalized_candidate_ids.append(saved_agent.candidate_id)
        return FinalizedCandidateSession(
            candidate_id=saved_agent.candidate_id,
            db_path=Path("screening.sqlite3"),
            summary_status="completed",
        )

    monkeypatch.setattr(ui_sidebar, "finalize_candidate_session", fake_finalize)

    app = AppTest.from_function(_run_ui_app)
    app.run()
    finish_button = next(
        button for button in app.button if button.label == "Finish screening"
    )
    finish_button.click().run()

    assert not app.exception
    assert finalized_candidate_ids == ["candidate-1"]


def test_dashboard_renders_candidates_without_anthropic(monkeypatch):
    candidates = [
        _stored_candidate(
            FakeUiAgent("eligible-1"),
            full_name="Elena Eligible",
            bot_label="eligible",
            hr_summary="Elena is ready for HR review.",
        ),
        _stored_candidate(
            FakeUiAgent("review-1"),
            full_name="Rafael Review",
            bot_label=None,
            hr_summary=None,
        ),
    ]

    monkeypatch.delenv(ANTHROPIC_API_KEY_ENV, raising=False)
    monkeypatch.setattr(ui_app, "load_dotenv", lambda: None)
    monkeypatch.setattr(ui_dashboard, "list_candidates", lambda *, db_path: candidates)
    monkeypatch.setattr(ui_analytics, "list_candidates", lambda *, db_path: candidates)
    monkeypatch.setattr(
        ui_state,
        "start_candidate_session",
        lambda: (_ for _ in ()).throw(AssertionError("chat should not start")),
    )

    app = AppTest.from_function(_run_ui_app)
    app.run()

    assert not app.exception
    expander_labels = [expander.label for expander in app.expander]
    assert any("Elena Eligible · Eligible" in label for label in expander_labels)
    assert any("Rafael Review · Needs review" in label for label in expander_labels)
    assert any(ANTHROPIC_API_KEY_ENV in alert.value for alert in app.error)


def test_dashboard_groups_candidates_by_triage_with_needs_review_fallback():
    candidates = [
        _stored_candidate(FakeUiAgent("eligible-1"), bot_label="eligible"),
        _stored_candidate(FakeUiAgent("not-eligible-1"), bot_label="not_eligible"),
        _stored_candidate(FakeUiAgent("review-1"), bot_label=None),
    ]

    grouped = dashboard_logic.group_candidates_by_triage(candidates)

    assert [candidate.id for candidate in grouped["eligible"]] == ["eligible-1"]
    assert [candidate.id for candidate in grouped["not_eligible"]] == ["not-eligible-1"]
    assert [candidate.id for candidate in grouped["needs_review"]] == ["review-1"]


def test_candidate_finality_comes_from_lifecycle_status_not_summary_status():
    completed_without_summary = _stored_candidate(
        FakeUiAgent("completed-1"),
        status="completed",
        summary_status=None,
    )
    active_with_summary = _stored_candidate(
        FakeUiAgent("active-1"),
        status="active",
        summary_status="completed",
    )
    disqualified_candidate = _stored_candidate(
        FakeUiAgent("disqualified-1"),
        status="disqualified",
        summary_status=None,
    )

    assert ui_state.is_finalized(completed_without_summary)
    assert ui_state.is_finalized(disqualified_candidate)
    assert not ui_state.is_finalized(active_with_summary)


def test_dashboard_filters_candidates_by_triage_city_and_search():
    madrid_candidate = _stored_candidate(
        FakeUiAgent("madrid-1"),
        full_name="Maria Madrid",
        city_zone="Madrid",
        bot_label="eligible",
    )
    barcelona_candidate = _stored_candidate(
        FakeUiAgent("barcelona-1"),
        full_name="Bruno Barcelona",
        city_zone="Barcelona",
        bot_label="needs_review",
    )

    filtered = dashboard_logic.filter_candidates(
        [madrid_candidate, barcelona_candidate],
        dashboard_logic.DashboardFilters(
            triage="Eligible",
            status="active",
            city_zone="Madrid",
            search="maria",
        ),
    )

    assert filtered == [madrid_candidate]


def test_dashboard_open_chat_resumes_candidate(monkeypatch):
    candidate = _stored_candidate(
        FakeUiAgent("dashboard-1"),
        full_name="Diana Dashboard",
        bot_label="eligible",
    )
    resumed_candidate_ids: list[str] = []

    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "test-key")
    monkeypatch.setattr(ui_dashboard, "list_candidates", lambda *, db_path: [candidate])
    monkeypatch.setattr(ui_analytics, "list_candidates", lambda *, db_path: [])
    monkeypatch.setattr(
        ui_state, "start_candidate_session", lambda: FakeUiAgent("current")
    )

    def fake_resume(candidate_id: str):
        resumed_candidate_ids.append(candidate_id)
        return FakeUiAgent(candidate_id)

    monkeypatch.setattr(ui_state, "resume_candidate_session", fake_resume)

    app = AppTest.from_function(_run_ui_app)
    app.run()
    open_button = next(button for button in app.button if button.label == "Open chat")
    open_button.click().run()

    assert not app.exception
    assert resumed_candidate_ids == ["dashboard-1"]


def test_analytics_kpis_handle_empty_candidates_and_duration_stats():
    assert analytics_logic.analytics_kpis([]) == analytics_logic.AnalyticsKpis(
        started=0,
        completed=0,
        completion_rate=0,
        qualified=0,
        needs_review=0,
        average_duration_minutes=None,
        median_duration_minutes=None,
    )

    candidates = [
        _stored_candidate(
            FakeUiAgent("eligible-1"),
            city_zone="Madrid",
            status="completed",
            bot_label="eligible",
            started_at="2026-01-01T10:00:00+00:00",
            completed_at="2026-01-01T10:10:00+00:00",
        ),
        _stored_candidate(
            FakeUiAgent("review-1"),
            city_zone="Barcelona",
            status="completed",
            bot_label="needs_review",
            started_at="2026-01-01T10:00:00+00:00",
            completed_at="2026-01-01T10:30:00+00:00",
        ),
        _stored_candidate(
            FakeUiAgent("active-1"),
            city_zone=None,
            status="active",
            bot_label=None,
        ),
    ]

    kpis = analytics_logic.analytics_kpis(candidates)

    assert kpis.started == 3
    assert kpis.completed == 2
    assert kpis.completion_rate == 2 / 3
    assert kpis.qualified == 1
    assert kpis.needs_review == 2
    assert kpis.average_duration_minutes == 20
    assert kpis.median_duration_minutes == 20


def test_dropoff_stage_prioritizes_clarification_then_missing_fields():
    clarification_candidate = _stored_candidate(
        FakeUiAgent("clarify-1"),
        city_zone="Madrid",
        status="active",
        drivers_license="Unknown",
    )
    missing_candidate = _stored_candidate(
        FakeUiAgent("missing-1"),
        city_zone=None,
        status="active",
    )
    completed_candidate = _stored_candidate(
        FakeUiAgent("complete-1"),
        city_zone="Madrid",
        status="completed",
        bot_label="eligible",
        completed_at="2026-01-01T00:10:00+00:00",
    )

    assert analytics_logic.dropoff_stage(clarification_candidate) == "drivers_license"
    assert analytics_logic.dropoff_stage(missing_candidate) == "city_zone"
    assert analytics_logic.dropoff_stage(completed_candidate) is None


def test_city_distribution_counts_known_and_unknown_cities():
    candidates = [
        _stored_candidate(
            FakeUiAgent("madrid-eligible"),
            city_zone="Madrid",
            bot_label="eligible",
        ),
        _stored_candidate(
            FakeUiAgent("madrid-review"),
            city_zone="Madrid",
            bot_label="needs_review",
        ),
        _stored_candidate(FakeUiAgent("unknown-city"), city_zone=None),
    ]

    rows = analytics_logic.city_distribution_rows(candidates)
    rows_by_city = {str(row["city"]): row for row in rows}

    assert rows_by_city["Madrid"]["count"] == 2
    assert rows_by_city["Madrid"]["eligible_share"] == 0.5
    assert rows_by_city["Madrid"]["lat"] == ui_constants.CITY_COORDINATES["Madrid"][0]
    assert rows_by_city["Madrid"]["size"] == ui_constants.MAP_BUBBLE_SIZE_SCALE * 2
    assert rows_by_city["Unknown"]["count"] == 1
    assert rows_by_city["Unknown"]["lat"] is None


def test_chart_rows_rank_largest_values_first():
    rows: list[dict[str, object]] = [
        {"stage": "Schedule", "candidates": 1},
        {"stage": "Driver license", "candidates": 4},
        {"stage": "Full name", "candidates": 2},
    ]

    ranked = analytics_logic.ranked_chart_rows(
        rows,
        value="candidates",
        category="stage",
    )

    assert [row["stage"] for row in ranked] == [
        "Driver license",
        "Full name",
        "Schedule",
    ]


def test_funnel_rows_preserve_required_field_order():
    rows = analytics_logic.funnel_completion_rows(
        [_stored_candidate(FakeUiAgent("candidate-1"))]
    )

    assert [row["stage"] for row in rows] == analytics_logic.funnel_stage_order()


def test_city_summary_rows_rank_largest_cities_first():
    rows: list[dict[str, object]] = [
        {"city": "Barcelona", "count": 1, "eligible_count": 0, "eligible_share": 0},
        {"city": "Madrid", "count": 3, "eligible_count": 2, "eligible_share": 2 / 3},
    ]

    summary = analytics_logic.city_summary_rows(rows)

    assert [row["City"] for row in summary] == ["Madrid", "Barcelona"]
    assert summary[0]["Candidates"] == 3
    assert summary[0]["Eligible"] == 2
    assert summary[0]["Eligible share"] == "67%"


def test_stale_active_candidates_ignore_recent_and_completed_candidates():
    candidates = [
        _stored_candidate(
            FakeUiAgent("old-active"),
            status="active",
            updated_at="2026-01-01T00:00:00+00:00",
        ),
        _stored_candidate(
            FakeUiAgent("recent-active"),
            status="active",
            updated_at="2026-01-02T11:00:00+00:00",
        ),
        _stored_candidate(
            FakeUiAgent("old-completed"),
            status="completed",
            updated_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:10:00+00:00",
        ),
    ]

    stale = analytics_logic.stale_active_candidates(
        candidates,
        now=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
    )

    assert [candidate.id for candidate in stale] == ["old-active"]


def test_analytics_tab_renders_with_injected_candidates_without_anthropic(monkeypatch):
    candidates = [
        _stored_candidate(
            FakeUiAgent("eligible-1"),
            full_name="Elena Eligible",
            city_zone="Madrid",
            status="completed",
            bot_label="eligible",
            hr_summary="Elena is ready for HR review.",
            completed_at="2026-01-01T00:10:00+00:00",
        ),
        _stored_candidate(
            FakeUiAgent("active-1"),
            full_name="Rafael Review",
            city_zone=None,
            status="active",
            bot_label=None,
        ),
    ]

    monkeypatch.delenv(ANTHROPIC_API_KEY_ENV, raising=False)
    monkeypatch.setattr(ui_app, "load_dotenv", lambda: None)
    monkeypatch.setattr(ui_dashboard, "list_candidates", lambda *, db_path: [])
    monkeypatch.setattr(ui_analytics, "list_candidates", lambda *, db_path: candidates)

    app = AppTest.from_function(_run_ui_app)
    app.run()

    assert not app.exception
    assert any("Screening funnel" in markdown.value for markdown in app.markdown)
    assert any("City distribution" in markdown.value for markdown in app.markdown)
    assert any("Stale active candidates" in markdown.value for markdown in app.markdown)


def _stored_candidate(
    agent: FakeUiAgent,
    *,
    full_name: str = "Maria Garcia",
    city_zone: str | None = None,
    drivers_license: DriverLicense = "Yes",
    status: str = "active",
    bot_label: str | None = None,
    hr_summary: str | None = None,
    summary_status: str | None = None,
    started_at: str = "2026-01-01T00:00:00+00:00",
    updated_at: str = "2026-01-01T00:00:00+00:00",
    completed_at: str | None = None,
) -> StoredCandidate:
    profile = CandidateProfile(
        full_name=full_name,
        drivers_license=drivers_license,
        raw_city_zone=city_zone,
        city_zone=city_zone,
        city_zone_status="Matched" if city_zone is not None else None,
        availability="Full-time",
        preferred_schedule="Morning",
        prior_delivery_experience=DeliveryExperience(years=1, platform="Glovo"),
        start_date="tomorrow",
    )
    return StoredCandidate(
        id=agent.candidate_id,
        profile=profile,
        status=status,
        started_at=started_at,
        updated_at=updated_at,
        completed_at=completed_at,
        hr_summary=hr_summary,
        bot_label=bot_label,
        summary_status=summary_status,
        summary_model=None,
        summary_error=None,
        summary_generated_at=None,
    )


def _run_ui_app():
    import screening.ui.app as ui_app

    ui_app.render_app()
