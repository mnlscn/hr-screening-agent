import json
from functools import lru_cache
from importlib.resources import files
from typing import TypedDict

from anthropic.types import MessageParam


class ServiceAreasByCountry(TypedDict):
    Spain: list[str]
    Mexico: list[str]


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


@lru_cache(maxsize=1)
def load_service_areas() -> ServiceAreasByCountry:
    resource = files("screening.data").joinpath("service_areas.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return {
        "Spain": [str(city) for city in payload["Spain"]],
        "Mexico": [str(city) for city in payload["Mexico"]],
    }


@lru_cache(maxsize=1)
def load_service_area_names() -> tuple[str, ...]:
    service_areas = load_service_areas()
    return tuple([*service_areas["Spain"], *service_areas["Mexico"]])
