import os
import signal
from types import FrameType

from dotenv import load_dotenv

from screening.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV
from screening.session import (
    FinalizedCandidateSession,
    finalize_candidate_session,
    save_current_session as save_agent_session,
)
from screening.storage import create_candidate


def main() -> None:
    load_dotenv()
    candidate_id = create_candidate()
    agent = ChatAgent(
        api_key=os.getenv(ANTHROPIC_API_KEY_ENV),
        candidate_id=candidate_id,
    )

    def save_current_session() -> None:
        save_agent_session(agent)

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


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    main()
