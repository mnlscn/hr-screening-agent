"""ChatAgent: drives the screening conversation and extracts candidate profiles."""

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
from screening.domain.models import CandidateProfile
from screening.llm.extraction import CandidateExtractor
from screening.llm.prompts.agent import OPENING_MESSAGE, build_system_prompt
from screening.llm.utils import count_tokens


@dataclass(frozen=True)
class AgentStream:
    """Streamed assistant response plus memory-management flags.

    Attributes:
        chunks (Iterator[str]): Iterator yielding the assistant reply text in
            streamed chunks.
        memory_truncated (bool): True when older messages were dropped to fit
            the input token budget.
        input_too_large (bool): True when a single message exceeded the input
            token budget and could not be pruned.
    """

    chunks: Iterator[str]
    memory_truncated: bool = False
    input_too_large: bool = False


class ChatAgent:
    """Stateful screening agent driving the conversation and profile extraction.

    Holds the conversation transcript and the running candidate profile, calls
    the Anthropic API to stream assistant replies, and re-extracts the profile
    after each user turn.

    Attributes:
        client (Anthropic): Anthropic API client used for chat and extraction.
        extractor (CandidateExtractor): Profile extractor bound to the client.
        candidate_id (str | None): Identifier of the persisted candidate, if any.
        messages (list[MessageParam]): The conversation transcript.
        profile (CandidateProfile): The current extracted candidate profile.
        last_extraction_error (Exception | None): The most recent extraction
            error, or None when the last extraction succeeded.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        candidate_id: str | None = None,
        messages: list[MessageParam] | None = None,
        profile: CandidateProfile | None = None,
    ) -> None:
        """Initialize the agent and its Anthropic client.

        Args:
            api_key (str | None): Anthropic API key passed to the client.
            candidate_id (str | None): Identifier of an existing candidate, if
                resuming.
            messages (list[MessageParam] | None): Existing transcript to seed
                the conversation; defaults to empty.
            profile (CandidateProfile | None): Existing profile to resume from;
                defaults to a new empty profile.
        """
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
        """Build an agent from a persisted candidate's stored state.

        Args:
            candidate_id (str): Identifier of the candidate to load.
            api_key (str | None): Anthropic API key passed to the new agent.
            db_path (str | Path): Path to the screening database.

        Returns:
            ChatAgent: An agent seeded with the stored profile and transcript.

        Raises:
            ValueError: If no candidate exists for the given identifier.
        """
        from screening.persistence.storage import load_agent_state

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
        """Begin the conversation with Lucia's opening message.

        No-op when the transcript already has messages.

        Returns:
            AgentStream: A stream yielding the opening message, or an empty
                stream when the conversation already started.
        """
        if self.messages:
            return AgentStream(chunks=iter(()))

        self.messages.append({"role": "assistant", "content": OPENING_MESSAGE})
        self.last_extraction_error = None

        return AgentStream(chunks=iter([OPENING_MESSAGE]))

    def stream(self, user_input: str) -> AgentStream:
        """Append a user message and stream the assistant's reply.

        Prunes memory to the token budget, re-extracts the profile, and returns
        a stream of the assistant response.

        Args:
            user_input (str): The candidate's latest message.

        Returns:
            AgentStream: The streamed reply with memory-management flags set.
        """
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
        """Stream the assistant reply, rolling back state on failure.

        Yields response text as it arrives and appends the full reply to the
        transcript on success. On error, removes the trailing user message and
        restores the previous profile and extraction error before re-raising.

        Args:
            previous_profile (CandidateProfile | None): Profile to restore if
                streaming fails.
            previous_extraction_error (Exception | None): Extraction error to
                restore if streaming fails.

        Yields:
            str: Chunks of the assistant's reply text.

        Raises:
            Exception: Re-raises any error raised while streaming the response.
        """
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
        """Return the transcript adapted for the API's role ordering.

        When the transcript opens with an assistant message, prepends a
        synthetic user context message so the request starts with a user turn.

        Returns:
            list[MessageParam]: Messages ready to send to the chat API.
        """
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
        """Re-extract the candidate profile from the transcript.

        Updates ``profile`` on success. On failure, leaves the profile
        unchanged and records the error in ``last_extraction_error``.
        """
        self.last_extraction_error = None
        try:
            self.profile = self.extractor.extract(self.messages, self.profile)
        except Exception as error:
            self.last_extraction_error = error

    def _latest_user_message(self) -> str | None:
        """Return the text of the most recent user message.

        Returns:
            str | None: The latest user message text, or None when there is no
                user message or its content is not plain text.
        """
        for message in reversed(self.messages):
            if message["role"] == "user":
                content = message["content"]
                return content if isinstance(content, str) else None
        return None

    def _prune_memory(self) -> tuple[bool, bool]:
        """Drop oldest messages until the transcript fits the token budget.

        Removes messages from the front while the estimated token count exceeds
        ``MAX_INPUT_TOKENS``, stopping when only one message remains.

        Returns:
            tuple[bool, bool]: A pair ``(memory_truncated, input_too_large)``
                where the first flag is True when messages were dropped and the
                second is True when a single remaining message still exceeds the
                budget.
        """
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
