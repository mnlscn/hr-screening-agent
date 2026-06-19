"""LLM response helpers: token estimation and text extraction."""

from typing import TypeGuard

from anthropic.types import Message, MessageParam, TextBlock


def count_tokens(messages: list[MessageParam]) -> int:
    """Approximate the token count of a message list.

    Uses a rough characters-per-token heuristic so the memory-pruning backstop
    works without a tokenizer dependency. This is intentionally an estimate:
    pruning only fires on pathological input (see MAX_INPUT_TOKENS), so the
    budget never sits near a hard limit where precision would matter, and the
    estimate stays cheap and local (no per-turn network call). Non-string
    content counts as a single token.

    For an exact, model-specific count (e.g. cost reporting or a real ceiling),
    use the Anthropic API: ``client.messages.count_tokens(model=..., ...)``.
    Do not reach for ``tiktoken`` — it is OpenAI's tokenizer and miscounts
    Claude tokens, more so on the Spanish text this agent handles.

    Args:
        messages (list[MessageParam]): Conversation messages to measure.

    Returns:
        int: The estimated total token count.
    """
    total = 0
    for message in messages:
        content = message["content"]
        total += len(content) // 4 + 1 if isinstance(content, str) else 1
    return total


def extract_text(response: Message) -> str:
    """Concatenate the text of all text blocks in an Anthropic response.

    Args:
        response (Message): An Anthropic message response with a ``content``
            block list.

    Returns:
        str: The joined text of every block whose type is "text".
    """
    return "".join(block.text for block in response.content if _is_text_block(block))


def _is_text_block(block: object) -> TypeGuard[TextBlock]:
    """Return whether a response content block carries text."""
    return getattr(block, "type", None) == "text"
