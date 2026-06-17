import os

from dotenv import load_dotenv

from screening.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV
from screening.storage import append_candidate_session


def main() -> None:
    load_dotenv()
    agent = ChatAgent(api_key=os.getenv(ANTHROPIC_API_KEY_ENV))

    print("Chatbot ready! Type 'quit' or 'exit' to end the conversation.\n")
    print("Lucia: ", end="", flush=True)
    for chunk in agent.start().chunks:
        print(chunk, end="", flush=True)
    print("\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit"]:
            if any(message["role"] == "user" for message in agent.messages):
                try:
                    output_path = append_candidate_session(
                        profile=agent.profile,
                        transcript=agent.messages,
                    )
                    print(f"[System] Candidate session saved to {output_path}.")
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
        except Exception as error:
            print(f"[Error] {error}")
            continue


if __name__ == "__main__":
    main()
