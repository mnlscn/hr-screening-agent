"""LLM response helpers: token estimation, text extraction, and JSON parsing."""

import json

from anthropic.types import MessageParam


def count_tokens(messages: list[MessageParam]) -> int:
    """Approximate tokens so memory pruning works without a tokenizer dependency."""
    total = 0
    for message in messages:
        content = message["content"]
        total += len(content) // 4 + 1 if isinstance(content, str) else 1
    return total


def extract_text(response):
    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)
