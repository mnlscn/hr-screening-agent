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
