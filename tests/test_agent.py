from typing import Any

import pytest

import screening.agent as agent_module
from screening.prompt import OPENING_MESSAGE


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
