"""Tests for service-area loading and canonical name resolution."""

from screening.domain.service_areas import load_service_area_names, load_service_areas


def test_load_service_areas_returns_structured_records():
    areas = load_service_areas()

    # The client operates 45 service areas across Spain and Mexico.
    assert len(areas["Spain"]) + len(areas["Mexico"]) == 45
    assert {"Madrid", "Barcelona", "Bilbao"} <= set(areas["Spain"])
    assert "Guadalajara" in areas["Mexico"]


def test_load_service_area_names_returns_canonical_names():
    names = load_service_area_names()

    assert "Barcelona" in names
    assert "Ciudad de Mexico" in names
    assert "Guadalajara" in names
