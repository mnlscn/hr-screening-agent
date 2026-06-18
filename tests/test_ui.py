from pathlib import Path
from typing import cast

from anthropic.types import MessageParam
from streamlit.testing.v1 import AppTest

import screening.ui as ui
from screening.agent import AgentStream
from screening.models import CandidateProfile, DeliveryExperience
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

    rows = dict(ui._profile_rows(profile))

    assert rows["Full name"] == "Maria Garcia"
    assert rows["Experience"] == "2 years, Glovo"
    assert rows["Availability"] == "Full-time"


def test_render_app_uses_injected_agent_without_anthropic(monkeypatch):
    agent = FakeUiAgent()
    candidate = _stored_candidate(agent)

    monkeypatch.setenv(ui.ANTHROPIC_API_KEY_ENV, "test-key")
    monkeypatch.setattr(ui, "start_candidate_session", lambda: agent)
    monkeypatch.setattr(
        ui, "load_candidate", lambda candidate_id, *, db_path: candidate
    )

    app = AppTest.from_function(_run_ui_app)
    app.run()

    assert not app.exception
    assert app.title[0].value == "Lucia Screening"
    assert any("Hola" in markdown.value for markdown in app.markdown)


def test_chat_submit_streams_and_saves_with_injected_agent(monkeypatch):
    agent = FakeUiAgent()
    candidate = _stored_candidate(agent)
    saved_messages: list[list[MessageParam]] = []

    monkeypatch.setenv(ui.ANTHROPIC_API_KEY_ENV, "test-key")
    monkeypatch.setattr(ui, "start_candidate_session", lambda: agent)
    monkeypatch.setattr(
        ui, "load_candidate", lambda candidate_id, *, db_path: candidate
    )
    monkeypatch.setattr(
        ui,
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

    monkeypatch.setenv(ui.ANTHROPIC_API_KEY_ENV, "test-key")
    monkeypatch.setattr(ui, "start_candidate_session", lambda: agent)
    monkeypatch.setattr(
        ui, "load_candidate", lambda candidate_id, *, db_path: candidate
    )

    def fake_finalize(saved_agent):
        finalized_candidate_ids.append(saved_agent.candidate_id)
        return FinalizedCandidateSession(
            candidate_id=saved_agent.candidate_id,
            db_path=Path("screening.sqlite3"),
            summary_status="completed",
        )

    monkeypatch.setattr(ui, "finalize_candidate_session", fake_finalize)

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

    monkeypatch.delenv(ui.ANTHROPIC_API_KEY_ENV, raising=False)
    monkeypatch.setattr(ui, "load_dotenv", lambda: None)
    monkeypatch.setattr(ui, "list_candidates", lambda *, db_path: candidates)
    monkeypatch.setattr(
        ui,
        "start_candidate_session",
        lambda: (_ for _ in ()).throw(AssertionError("chat should not start")),
    )

    app = AppTest.from_function(_run_ui_app)
    app.run()

    assert not app.exception
    assert any("Elena Eligible" in markdown.value for markdown in app.markdown)
    assert any("Rafael Review" in markdown.value for markdown in app.markdown)
    assert any(ui.ANTHROPIC_API_KEY_ENV in alert.value for alert in app.error)


def test_dashboard_groups_candidates_by_triage_with_needs_review_fallback():
    candidates = [
        _stored_candidate(FakeUiAgent("eligible-1"), bot_label="eligible"),
        _stored_candidate(FakeUiAgent("not-eligible-1"), bot_label="not_eligible"),
        _stored_candidate(FakeUiAgent("review-1"), bot_label=None),
    ]

    grouped = ui._group_candidates_by_triage(candidates)

    assert [candidate.id for candidate in grouped["eligible"]] == ["eligible-1"]
    assert [candidate.id for candidate in grouped["not_eligible"]] == ["not-eligible-1"]
    assert [candidate.id for candidate in grouped["needs_review"]] == ["review-1"]


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

    filtered = ui._filter_candidates(
        [madrid_candidate, barcelona_candidate],
        ui.DashboardFilters(
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

    monkeypatch.setenv(ui.ANTHROPIC_API_KEY_ENV, "test-key")
    monkeypatch.setattr(ui, "list_candidates", lambda *, db_path: [candidate])
    monkeypatch.setattr(ui, "start_candidate_session", lambda: FakeUiAgent("current"))

    def fake_resume(candidate_id: str):
        resumed_candidate_ids.append(candidate_id)
        return FakeUiAgent(candidate_id)

    monkeypatch.setattr(ui, "resume_candidate_session", fake_resume)

    app = AppTest.from_function(_run_ui_app)
    app.run()
    open_button = next(button for button in app.button if button.label == "Open chat")
    open_button.click().run()

    assert not app.exception
    assert resumed_candidate_ids == ["dashboard-1"]


def _stored_candidate(
    agent: FakeUiAgent,
    *,
    full_name: str = "Maria Garcia",
    city_zone: str | None = None,
    bot_label: str | None = None,
    hr_summary: str | None = None,
    summary_status: str | None = None,
) -> StoredCandidate:
    profile = CandidateProfile(
        full_name=full_name,
        drivers_license="Yes",
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
        status="active",
        started_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        completed_at=None,
        hr_summary=hr_summary,
        bot_label=bot_label,
        summary_status=summary_status,
        summary_model=None,
        summary_error=None,
        summary_generated_at=None,
    )


def _run_ui_app():
    import screening.ui as ui_module

    ui_module.render_app()
