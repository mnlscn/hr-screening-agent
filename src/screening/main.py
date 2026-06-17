import os

from dotenv import load_dotenv

from screening.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV
from screening.storage import create_candidate, save_candidate_session


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

    print("Lucia: ", end="", flush=True)
    for chunk in agent.start().chunks:
        print(chunk, end="", flush=True)
    print("\n")
    save_current_session()

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit"]:
            try:
                saved_session = save_candidate_session(
                    profile=agent.profile,
                    transcript=agent.messages,
                    candidate_id=agent.candidate_id,
                )
                print(
                    "[System] Candidate "
                    f"{saved_session.candidate_id} saved to {saved_session.db_path}."
                )
            except Exception as error:
                print(f"[System] Could not save candidate session: {error}")
            print("Goodbye!")
            break

        try:
            response = agent.stream(user_input)

            if response.memory_truncated:
                print("[System] Conversation history truncated to fit memory budget.")

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


if __name__ == "__main__":
    main()
