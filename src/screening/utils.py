def count_tokens(messages):
    """Approximate tokens so memory pruning works without a tokenizer dependency."""
    return sum(len(message["content"]) // 4 + 1 for message in messages)


def extract_text(response):
    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )
