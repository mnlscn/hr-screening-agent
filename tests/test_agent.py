import unittest
from unittest.mock import patch

from screening.agent import ChatAgent
from screening.prompt import OPENING_MESSAGE


class ChatAgentTest(unittest.TestCase):
    @patch("screening.agent.Anthropic")
    def test_start_adds_opening_message_without_extraction(self, anthropic_class):
        agent = ChatAgent(api_key="test-key")

        response = agent.start()

        self.assertEqual("".join(response.chunks), OPENING_MESSAGE)
        self.assertEqual(
            agent.messages,
            [{"role": "assistant", "content": OPENING_MESSAGE}],
        )
        self.assertIsNone(agent.last_extraction_error)
        anthropic_class.assert_called_once_with(api_key="test-key")

    @patch("screening.agent.Anthropic")
    def test_messages_for_api_prefixes_context_when_transcript_starts_with_assistant(
        self, _anthropic_class
    ):
        agent = ChatAgent(api_key="test-key")
        agent.start()
        agent.messages.append({"role": "user", "content": "Si, podemos empezar"})

        messages = agent._messages_for_api()

        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user"])
        self.assertIn("already sent the first recruiting outreach", messages[0]["content"])
        self.assertEqual(messages[1]["content"], OPENING_MESSAGE)


if __name__ == "__main__":
    unittest.main()
