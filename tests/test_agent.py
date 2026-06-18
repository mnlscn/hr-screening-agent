"""Tests for ChatAgent: streaming, memory pruning, and profile extraction."""

from typing import Any
from typing import cast

import pytest
from anthropic.types import MessageParam

import screening.llm.agent as agent_module
from screening.domain.models import CandidateProfile
from screening.llm.prompts.agent import OPENING_MESSAGE
from screening.persistence.storage import save_candidate_session


class FakeStream:
    def __init__(self, chunks: list[str]):
        self.text_stream = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class RecordingMessages:
    def __init__(self):
        self.system_prompts: list[str] = []

    def stream(self, **kwargs):
        self.system_prompts.append(kwargs["system"])
        return FakeStream(["ok"])


class RecordingClient:
    def __init__(self):
        self.messages = RecordingMessages()


class AvailabilityExtractor:
    def extract(self, messages, current_profile):
        return current_profile.merge(CandidateProfile(availability="Full-time"))


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
    anthropic_calls: list[dict[str, str | None]],
):
    agent = agent_module.ChatAgent(
        api_key="test-key",
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
