import json
from functools import lru_cache
from importlib.resources import files
from typing import TypedDict


class ServiceAreasByCountry(TypedDict):
    Spain: list[str]
    Mexico: list[str]


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
