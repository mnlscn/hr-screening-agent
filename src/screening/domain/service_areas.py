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
def _load_payload() -> dict[str, dict[str, dict[str, float]]]:
    """Read and cache the raw service-areas JSON resource.

    Returns:
        dict[str, dict[str, dict[str, float]]]: Country to city to coordinate
            mapping, e.g. ``{"Spain": {"Madrid": {"lat": ..., "lon": ...}}}``.
    """
    resource = files("screening.data").joinpath("service_areas.json")
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_service_areas() -> ServiceAreasByCountry:
    """Load the bundled service-area cities grouped by country.

    The result is cached so the JSON resource is read only once.

    Returns:
        ServiceAreasByCountry: Supported city names keyed by country.
    """
    payload = _load_payload()
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


@lru_cache(maxsize=1)
def load_city_coordinates() -> dict[str, tuple[float, float]]:
    """Return ``(latitude, longitude)`` for every service-area city.

    Coordinates are flattened across all countries into a single mapping
    keyed by canonical city name. The result is cached.

    Returns:
        dict[str, tuple[float, float]]: City name to ``(lat, lon)`` mapping.
    """
    payload = _load_payload()
    coordinates: dict[str, tuple[float, float]] = {}
    for cities in payload.values():
        for city, coords in cities.items():
            coordinates[str(city)] = (float(coords["lat"]), float(coords["lon"]))
    return coordinates
