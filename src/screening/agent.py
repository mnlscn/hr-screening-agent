from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import MessageParam

from screening.config import (
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MODEL,
    SCREENING_DB_PATH,
)
from screening.extraction import CandidateExtractor
from screening.models import CandidateProfile
from screening.prompt import OPENING_MESSAGE, build_system_prompt
from screening.utils import count_tokens


@dataclass(frozen=True)
class AgentStream:
    chunks: Iterator[str]
    memory_truncated: bool = False
    input_too_large: bool = False


class ChatAgent:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        candidate_id: str | None = None,
        messages: list[MessageParam] | None = None,
        profile: CandidateProfile | None = None,
    ) -> None:
        self.client = Anthropic(api_key=api_key)
        self.extractor = CandidateExtractor(self.client)
        self.candidate_id = candidate_id
        self.messages: list[MessageParam] = list(messages or [])
        self.profile = profile or CandidateProfile()
        self.last_extraction_error: Exception | None = None

    @classmethod
    def from_candidate_id(
        cls,
        candidate_id: str,
        *,
        api_key: str | None = None,
        db_path: str | Path = SCREENING_DB_PATH,
    ) -> "ChatAgent":
        from screening.storage import load_agent_state

        state = load_agent_state(candidate_id, db_path=db_path)
        if state is None:
            raise ValueError(f"candidate not found: {candidate_id}")

        return cls(
            api_key=api_key,
            candidate_id=state.candidate_id,
            messages=state.messages,
            profile=state.profile,
        )

    def start(self) -> AgentStream:
        if self.messages:
            return AgentStream(chunks=iter(()))

        self.messages.append({"role": "assistant", "content": OPENING_MESSAGE})
        self.last_extraction_error = None

        return AgentStream(chunks=iter([OPENING_MESSAGE]))

    def stream(self, user_input: str) -> AgentStream:
        previous_profile = self.profile
        previous_extraction_error = self.last_extraction_error
        self.messages.append({"role": "user", "content": user_input})
        memory_truncated, input_too_large = self._prune_memory()
        self._extract_profile()

        return AgentStream(
            chunks=self._stream_response(previous_profile, previous_extraction_error),
            memory_truncated=memory_truncated,
            input_too_large=input_too_large,
        )

    def _stream_response(
        self,
        previous_profile: CandidateProfile | None = None,
        previous_extraction_error: Exception | None = None,
    ) -> Iterator[str]:
        chunks = []

        try:
            with self.client.messages.stream(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=build_system_prompt(
                    self.profile,
                    latest_user_message=self._latest_user_message(),
                ),
                messages=self._messages_for_api(),
            ) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
                    yield text
        except Exception:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            if previous_profile is not None:
                self.profile = previous_profile
                self.last_extraction_error = previous_extraction_error
            raise

        self.messages.append({"role": "assistant", "content": "".join(chunks)})

    def _messages_for_api(self) -> list[MessageParam]:
        if not self.messages or self.messages[0]["role"] != "assistant":
            return self.messages

        return [
            {
                "role": "user",
                "content": (
                    "Context: Lucia already sent the first recruiting outreach "
                    "message to the candidate. Continue the screening chat using "
                    "the transcript below."
                ),
            },
            *self.messages,
        ]

    def _extract_profile(self) -> None:
        self.last_extraction_error = None
        try:
            self.profile = self.extractor.extract(self.messages, self.profile)
        except Exception as error:
            self.last_extraction_error = error

    def _latest_user_message(self) -> str | None:
        for message in reversed(self.messages):
            if message["role"] == "user":
                content = message["content"]
                return content if isinstance(content, str) else None
        return None

    def _prune_memory(self) -> tuple[bool, bool]:
        memory_truncated = False
        input_too_large = False

        while count_tokens(self.messages) > MAX_INPUT_TOKENS:
            if len(self.messages) > 1:
                self.messages.pop(0)
                memory_truncated = True
            else:
                input_too_large = True
                break

        return memory_truncated, input_too_large
