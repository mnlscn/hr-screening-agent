"""Tests for ChatAgent: streaming, memory pruning, and profile extraction."""

import json
from pathlib import Path
from typing import Any
from typing import cast

import pytest
from anthropic.types import MessageParam

import screening.llm.agent as agent_module
from screening.domain.models import CandidateProfile
from screening.llm.prompts.agent import OPENING_MESSAGE
from screening.observability import configure_logging
from screening.persistence.storage import save_candidate_session


class FakeStream:
    def __init__(self, chunks: list[str]):
        self.text_stream = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FailingStream:
    text_stream = iter(())

    def __enter__(self):
        self.text_stream = self._raise_on_iter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def _raise_on_iter(self):
        raise RuntimeError("stream exploded")
        yield ""


class RecordingMessages:
    def __init__(self):
        self.system_prompts: list[str] = []

    def stream(self, **kwargs):
        self.system_prompts.append(kwargs["system"])
        return FakeStream(["ok"])


class RecordingClient:
    def __init__(self):
        self.messages = RecordingMessages()


class FailingMessages:
    def stream(self, **kwargs):
        return FailingStream()


class FailingClient:
    def __init__(self):
        self.messages = FailingMessages()


class AvailabilityExtractor:
    def extract(self, messages, current_profile):
        return current_profile.merge(CandidateProfile(availability="Full-time"))


class FailingExtractor:
    def extract(self, messages, current_profile):
        raise ValueError("extract boom")


@pytest.fixture
def anthropic_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str | None]]:
    calls: list[dict[str, str | None]] = []

    def fake_anthropic(*, api_key: str | None = None, **_: object) -> Any:
        calls.append({"api_key": api_key})
        return object()

    monkeypatch.setattr(
        agent_module,
        "Anthropic",
        fake_anthropic,
    )
    return calls


def test_start_adds_opening_message_without_extraction(
    anthropic_calls: list[dict[str, str | None]],
):
    agent = agent_module.ChatAgent(api_key="test-key")

    response = agent.start()

    assert "".join(response.chunks) == OPENING_MESSAGE
    assert agent.messages == [{"role": "assistant", "content": OPENING_MESSAGE}]
    assert agent.last_extraction_error is None
    assert anthropic_calls == [{"api_key": "test-key"}]


def test_messages_for_api_prefixes_context_when_transcript_starts_with_assistant(
    anthropic_calls: list[dict[str, str | None]],
):
    agent = agent_module.ChatAgent(api_key="test-key")
    agent.start()
    agent.messages.append({"role": "user", "content": "Si, podemos empezar"})

    messages = agent._messages_for_api()

    assert anthropic_calls == [{"api_key": "test-key"}]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert "already sent the first recruiting outreach" in messages[0]["content"]
    assert messages[1]["content"] == OPENING_MESSAGE


def test_from_candidate_id_reconstructs_agent_state(
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

    agent = agent_module.ChatAgent.from_candidate_id(
        saved.candidate_id,
        api_key="test-key",
        db_path=db_path,
    )

    assert agent.candidate_id == saved.candidate_id
    assert agent.profile.model_dump(mode="json") == profile.model_dump(mode="json")
    assert agent.messages == messages
    assert anthropic_calls == [{"api_key": "test-key"}]


def test_stream_extracts_latest_user_answer_before_building_prompt(
    tmp_path,
    anthropic_calls: list[dict[str, str | None]],
):
    log_path = _configure_test_logging(tmp_path)
    agent = agent_module.ChatAgent(
        api_key="test-key",
        candidate_id="candidate-1",
        profile=CandidateProfile(
            full_name="Elena Diaz Vicuna",
            drivers_license="Yes",
            raw_city_zone="Madrid",
            city_zone="Madrid",
            city_zone_status="Matched",
            start_date="September",
        ),
    )
    client = RecordingClient()
    agent.client = cast(Any, client)
    agent.extractor = cast(Any, AvailabilityExtractor())

    response = agent.stream("si")

    assert "".join(response.chunks) == "ok"
    assert agent.profile.availability == "Full-time"
    assert (
        '"next_field_to_collect": "preferred_schedule"'
        in (client.messages.system_prompts[0])
    )
    assert anthropic_calls == [{"api_key": "test-key"}]
    events = _log_events(log_path)
    assert "extraction_completed" in events
    assert "chat_stream_completed" in events


def test_stream_failure_logs_rollback_without_message_text(
    tmp_path,
    anthropic_calls: list[dict[str, str | None]],
):
    log_path = _configure_test_logging(tmp_path)
    agent = agent_module.ChatAgent(api_key="test-key", candidate_id="candidate-1")
    agent.client = cast(Any, FailingClient())
    agent.extractor = cast(Any, AvailabilityExtractor())

    response = agent.stream("Soy Secret Candidate")

    with pytest.raises(RuntimeError, match="stream exploded"):
        "".join(response.chunks)

    assert agent.messages == []
    assert agent.profile.availability is None
    records = _read_log_records(log_path)
    failure_record = _record_for_event(records, "chat_stream_failed")
    assert failure_record["extra"]["rollback_applied"] is True
    assert "Secret Candidate" not in _log_text(log_path)


def test_extraction_failure_is_logged_without_message_text(
    tmp_path,
    anthropic_calls: list[dict[str, str | None]],
):
    log_path = _configure_test_logging(tmp_path)
    agent = agent_module.ChatAgent(api_key="test-key", candidate_id="candidate-1")
    agent.client = cast(Any, RecordingClient())
    agent.extractor = cast(Any, FailingExtractor())

    response = agent.stream("Soy Secret Candidate")

    assert "".join(response.chunks) == "ok"
    assert isinstance(agent.last_extraction_error, ValueError)
    events = _log_events(log_path)
    assert "extraction_failed" in events
    assert "chat_stream_completed" in events
    assert "Secret Candidate" not in _log_text(log_path)


def _configure_test_logging(tmp_path: Path) -> Path:
    log_path = tmp_path / "screening.jsonl"
    configure_logging(log_path=log_path, include_stderr=False, force=True)
    return log_path


def _read_log_records(log_path: Path) -> list[dict]:
    return [
        json.loads(line)["record"]
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]


def _log_events(log_path: Path) -> set[str]:
    return {
        str(record["extra"]["event"])
        for record in _read_log_records(log_path)
        if "event" in record["extra"]
    }


def _record_for_event(records: list[dict], event: str) -> dict:
    return next(record for record in records if record["extra"].get("event") == event)


def _log_text(log_path: Path) -> str:
    return log_path.read_text(encoding="utf-8")
