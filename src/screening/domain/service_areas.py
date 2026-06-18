"""Loader for the service-areas reference data bundled with the package."""

import json
from functools import lru_cache
from importlib.resources import files
from typing import TypedDict


class ServiceAreasByCountry(TypedDict):
    """Service-area city names grouped by supported country.

    Attributes:
        Spain (list[str]): Canonical city names supported in Spain.
        Mexico (list[str]): Canonical city names supported in Mexico.
    """

    Spain: list[str]
    Mexico: list[str]


@lru_cache(maxsize=1)
def load_service_areas() -> ServiceAreasByCountry:
    """Load the bundled service-area cities grouped by country.

    The result is cached so the JSON resource is read only once.

    Returns:
        ServiceAreasByCountry: Supported city names keyed by country.
    """
    resource = files("screening.data").joinpath("service_areas.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return {
        "Spain": [str(city) for city in payload["Spain"]],
        "Mexico": [str(city) for city in payload["Mexico"]],
    }


@lru_cache(maxsize=1)
def load_service_area_names() -> tuple[str, ...]:
    """Return all supported city names as a single flat tuple.

    Combines the Spain and Mexico city lists into one collection. The result
    is cached.

    Returns:
        tuple[str, ...]: Every canonical service-area city name.
    """
    service_areas = load_service_areas()
    return tuple([*service_areas["Spain"], *service_areas["Mexico"]])
