import os
import signal
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from dotenv import load_dotenv

from screening.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV, SCREENING_DB_PATH, SUMMARY_MODEL
from screening.storage import (
    COMPLETED_STATUS,
    SavedCandidateSession,
    create_candidate,
    mark_candidate_summary_pending,
    save_candidate_session,
    save_candidate_summary,
    save_candidate_summary_failure,
)
from screening.summary import CandidateSummarizer


@dataclass(frozen=True)
class FinalizedCandidateSession:
    candidate_id: str
    db_path: Path
    summary_status: str


def main() -> None:
    load_dotenv()
    candidate_id = create_candidate()
    agent = ChatAgent(
        api_key=os.getenv(ANTHROPIC_API_KEY_ENV),
        candidate_id=candidate_id,
    )

    def save_current_session() -> None:
        save_candidate_session(
            profile=agent.profile,
            transcript=agent.messages,
            candidate_id=agent.candidate_id,
        )

    def finalize_current_session() -> FinalizedCandidateSession | None:
        try:
            finalized_session = finalize_candidate_session(agent)
            print(
                "[System] Candidate "
                f"{finalized_session.candidate_id} saved to "
                f"{finalized_session.db_path}. Summary status: "
                f"{finalized_session.summary_status}."
            )
            return finalized_session
        except Exception as error:
            print(f"[System] Could not finalize candidate session: {error}")
            return None

    print("Lucia: ", end="", flush=True)
    for chunk in agent.start().chunks:
        print(chunk, end="", flush=True)
    print("\n")
    save_current_session()

    previous_sigterm_handler = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        while True:
            user_input = input("You: ")

            if user_input.strip().lower() in ["quit", "exit"]:
                finalize_current_session()
                print("Goodbye!")
                break

            try:
                response = agent.stream(user_input)

                if response.memory_truncated:
                    print(
                        "[System] Conversation history truncated to fit memory budget."
                    )

                if response.input_too_large:
                    print("[System] Warning: Single message exceeds token limit.")

                print("Lucia: ", end="", flush=True)
                for chunk in response.chunks:
                    print(chunk, end="", flush=True)
                print("\n")

                if agent.last_extraction_error:
                    print(
                        "[System] Candidate extraction failed. "
                        "The conversation can continue."
                    )

                save_current_session()
            except Exception as error:
                print(f"[Error] {error}")
                continue
    except KeyboardInterrupt:
        print("\n[System] Finalizing candidate before exit...")
        finalize_current_session()
        print("Goodbye!")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


def finalize_candidate_session(
    agent: ChatAgent,
    *,
    db_path: str | Path = SCREENING_DB_PATH,
    summarizer: CandidateSummarizer | None = None,
) -> FinalizedCandidateSession:
    saved_session = save_candidate_session(
        profile=agent.profile,
        transcript=agent.messages,
        candidate_id=agent.candidate_id,
        db_path=db_path,
        status=COMPLETED_STATUS,
    )
    summary_status = _generate_and_store_summary(
        agent,
        saved_session,
        summarizer=summarizer,
    )
    return FinalizedCandidateSession(
        candidate_id=saved_session.candidate_id,
        db_path=saved_session.db_path,
        summary_status=summary_status,
    )


def _generate_and_store_summary(
    agent: ChatAgent,
    saved_session: SavedCandidateSession,
    *,
    summarizer: CandidateSummarizer | None = None,
) -> str:
    mark_candidate_summary_pending(
        saved_session.candidate_id,
        db_path=saved_session.db_path,
        model=SUMMARY_MODEL,
    )
    summarizer = summarizer or CandidateSummarizer(agent.client)

    try:
        summary = summarizer.summarize(
            agent.profile,
            agent.messages,
            extraction_failed=agent.last_extraction_error is not None,
        )
    except Exception as error:
        save_candidate_summary_failure(
            saved_session.candidate_id,
            error=str(error),
            model=SUMMARY_MODEL,
            db_path=saved_session.db_path,
        )
        return "failed"

    save_candidate_summary(
        saved_session.candidate_id,
        hr_summary=summary.hr_summary,
        bot_label=summary.bot_label,
        model=summary.model,
        db_path=saved_session.db_path,
    )
    return "completed"


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    main()
