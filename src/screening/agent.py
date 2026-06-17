from dataclasses import dataclass
from collections.abc import Iterator

from anthropic import Anthropic

from screening.config import MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, MODEL
from screening.extraction import CandidateExtractor
from screening.models import CandidateProfile
from screening.prompt import SYSTEM_PROMPT
from screening.utils import count_tokens


@dataclass(frozen=True)
class AgentStream:
    chunks: Iterator[str]
    memory_truncated: bool = False
    input_too_large: bool = False


class ChatAgent:
    def __init__(self, api_key: str | None = None) -> None:
        self.client = Anthropic(api_key=api_key)
        self.extractor = CandidateExtractor(self.client)
        self.messages = []
        self.profile = CandidateProfile()
        self.last_extraction_error: Exception | None = None

    def stream(self, user_input: str) -> AgentStream:
        self.messages.append({"role": "user", "content": user_input})
        memory_truncated, input_too_large = self._prune_memory()

        return AgentStream(
            chunks=self._stream_response(),
            memory_truncated=memory_truncated,
            input_too_large=input_too_large,
        )

    def _stream_response(self) -> Iterator[str]:
        chunks = []

        try:
            with self.client.messages.stream(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=self.messages,
            ) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
                    yield text
        except Exception:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            raise

        self.messages.append({"role": "assistant", "content": "".join(chunks)})
        self._extract_profile()

    def _extract_profile(self) -> None:
        self.last_extraction_error = None
        try:
            self.profile = self.extractor.extract(self.messages, self.profile)
        except Exception as error:
            self.last_extraction_error = error

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
