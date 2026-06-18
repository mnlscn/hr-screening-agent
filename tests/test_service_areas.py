"""Tests for service-area loading and canonical name resolution."""

from screening.domain.service_areas import load_service_area_names, load_service_areas


def test_load_service_areas_returns_structured_records():
    areas = load_service_areas()

    assert areas["Spain"] == [
        "Madrid",
        "Barcelona",
        "Valencia",
        "Sevilla",
        "Malaga",
        "Zaragoza",
        "Bilbao",
    ]
    assert "Guadalajara" in areas["Mexico"]


def test_load_service_area_names_returns_canonical_names():
    names = load_service_area_names()

    assert "Barcelona" in names
    assert "Ciudad de Mexico" in names
    assert "Guadalajara" in names
