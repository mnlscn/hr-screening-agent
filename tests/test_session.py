"""Tests for application session workflows: start, save, resume, and finalize."""

from typing import Any, cast

import pytest
from anthropic.types import MessageParam

import screening.llm.agent as agent_module
from screening.llm.agent import ChatAgent
from screening.domain.models import CandidateProfile, DeliveryExperience
from screening.llm.prompts.agent import OPENING_MESSAGE
from screening.application.session import (
    finalize_candidate_session,
    resume_candidate_session,
    save_current_session,
    start_candidate_session,
)
from screening.persistence.storage import (
    create_candidate,
    load_agent_state,
    load_candidate,
    load_candidate_messages,
    save_candidate_session,
)
from screening.llm.summary import CandidateSummary


class FakeAgent:
    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        self.profile = CandidateProfile(
            full_name="Maria Garcia",
            drivers_license="Yes",
            raw_city_zone="Madrid",
            city_zone="Madrid",
            city_zone_status="Matched",
            conversation_language="Spanish",
            availability="Full-time",
            preferred_schedule="Morning",
            prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
            start_date="next Monday",
        )
        self.messages = [
            cast(MessageParam, {"role": "assistant", "content": "Hola"}),
            cast(MessageParam, {"role": "user", "content": "Soy Maria Garcia"}),
        ]
        self.client = object()
        self.last_extraction_error = None


class FakeStream:
    def __init__(self, chunks: list[str]):
        self.text_stream = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class RecordingMessages:
    def stream(self, **kwargs):
        return FakeStream(["ok"])


class RecordingClient:
    def __init__(self):
        self.messages = RecordingMessages()


class AvailabilityExtractor:
    def extract(self, messages, current_profile):
        return current_profile.merge(CandidateProfile(availability="Full-time"))


class FakeSummarizer:
    def __init__(self):
        self.seen_messages: list[MessageParam] | None = None

    def summarize(self, profile, transcript, *, extraction_failed=False):
        self.seen_messages = list(transcript)
        return CandidateSummary(
            bot_label="eligible",
            hr_summary="Maria is complete, polite, and ready for HR review.",
        )


class FailingSummarizer:
    def summarize(self, profile, transcript, *, extraction_failed=False):
        raise ValueError("bad summary JSON")


@pytest.fixture
def anthropic_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str | None]]:
    calls: list[dict[str, str | None]] = []

    def fake_anthropic(*, api_key: str | None = None) -> Any:
        calls.append({"api_key": api_key})
        return object()

    monkeypatch.setattr(
        agent_module,
        "Anthropic",
        fake_anthropic,
    )
    return calls


def test_start_candidate_session_saves_opening_message(
    tmp_path,
    anthropic_calls: list[dict[str, str | None]],
):
    db_path = tmp_path / "screening.sqlite3"

    agent = start_candidate_session(api_key="test-key", db_path=db_path)

    assert agent.candidate_id is not None
    assert agent.messages == [{"role": "assistant", "content": OPENING_MESSAGE}]
    assert load_candidate_messages(agent.candidate_id, db_path=db_path) == [
        {"role": "assistant", "content": OPENING_MESSAGE}
    ]
    assert anthropic_calls == [{"api_key": "test-key"}]


def test_resume_candidate_session_reconstructs_agent_state(
    tmp_path,
    anthropic_calls: list[dict[str, str | None]],
):
    db_path = tmp_path / "screening.sqlite3"
    profile = CandidateProfile(full_name="Luis Perez")
    messages = [
        cast(MessageParam, {"role": "assistant", "content": "Hola"}),
        cast(MessageParam, {"role": "user", "content": "Soy Luis Perez"}),
    ]
    saved = save_candidate_session(profile, messages, db_path=db_path)

    agent = resume_candidate_session(
        saved.candidate_id,
        api_key="test-key",
        db_path=db_path,
    )

    assert agent.candidate_id == saved.candidate_id
    assert agent.profile.model_dump(mode="json") == profile.model_dump(mode="json")
    assert agent.messages == messages
    assert anthropic_calls == [{"api_key": "test-key"}]


def test_save_current_session_persists_streamed_response(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    candidate_id = create_candidate(db_path=db_path)
    agent = ChatAgent(api_key="test-key", candidate_id=candidate_id)
    agent.client = cast(Any, RecordingClient())
    agent.extractor = cast(Any, AvailabilityExtractor())

    response = agent.stream("si")
    assert "".join(response.chunks) == "ok"

    saved = save_current_session(agent, db_path=db_path)
    state = load_agent_state(saved.candidate_id, db_path=db_path)

    assert state is not None
    assert state.status == "active"
    assert state.profile.availability == "Full-time"
    assert state.messages == [
        {"role": "user", "content": "si"},
        {"role": "assistant", "content": "ok"},
    ]


def test_finalize_candidate_session_saves_before_summarizing(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    candidate_id = create_candidate(db_path=db_path)
    agent = FakeAgent(candidate_id)
    summarizer = FakeSummarizer()

    finalized = finalize_candidate_session(
        cast(Any, agent),
        db_path=db_path,
        summarizer=cast(Any, summarizer),
    )
    loaded = load_candidate(candidate_id, db_path=db_path)

    assert finalized.summary_status == "completed"
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.bot_label == "eligible"
    assert loaded.summary_status == "completed"
    assert loaded.hr_summary == "Maria is complete, polite, and ready for HR review."
    assert load_candidate_messages(candidate_id, db_path=db_path) == agent.messages
    assert summarizer.seen_messages == agent.messages


def test_finalize_candidate_session_records_summary_failure(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    candidate_id = create_candidate(db_path=db_path)
    agent = FakeAgent(candidate_id)

    finalized = finalize_candidate_session(
        cast(Any, agent),
        db_path=db_path,
        summarizer=cast(Any, FailingSummarizer()),
    )
    loaded = load_candidate(candidate_id, db_path=db_path)

    assert finalized.summary_status == "failed"
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.bot_label == "needs_review"
    assert loaded.summary_status == "failed"
    assert loaded.summary_error == "bad summary JSON"
