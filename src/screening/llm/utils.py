"""LLM response helpers: token estimation, text extraction, and JSON parsing."""

import json

from anthropic.types import MessageParam


def count_tokens(messages: list[MessageParam]) -> int:
    """Approximate the token count of a message list.

    Uses a rough characters-per-token heuristic so memory pruning works
    without a tokenizer dependency. Non-string content counts as a single
    token.

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


def extract_text(response):
    """Concatenate the text of all text blocks in an Anthropic response.

    Args:
        response: An Anthropic message response with a ``content`` block list.

    Returns:
        str: The joined text of every block whose type is "text".
    """
    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, stripping code fences.

    Removes a surrounding Markdown code fence when present before parsing.

    Args:
        text (str): Raw model output expected to contain a JSON object.

    Returns:
        dict: The parsed JSON object.

    Raises:
        json.JSONDecodeError: If the cleaned text is not valid JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)
