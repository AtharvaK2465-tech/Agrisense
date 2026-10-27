"""
Regression testing across required regional test scenarios:
Akola, Amravati, Nagpur, Indore, Barshi, Khopoli, and Wadhode.
Verifies spatial boundary handling, district resolution, crop calendar alignment,
and Maharashtra boundary constraint behavior.
"""
import pytest
from datetime import date
from utils.geo import validate_polygon, point_radius_to_geojson
from utils.districts import locate_district, season_for_date, plausible_crops
import utils.config as config


REGRESSION_SCENARIOS = {
    "Akola": {
        "lat": 20.70,
        "lon": 77.00,
        "expected_district": "Akola",
        "season": "kharif",
        "expected_crops": ["cotton", "soybean", "tur"],
        "in_maharashtra": True,
    },
    "Amravati": {
        "lat": 20.93,
        "lon": 77.75,
        "expected_district": "Amravati",
        "season": "kharif",
        "expected_crops": ["cotton", "soybean", "tur"],
        "in_maharashtra": True,
    },
    "Nagpur": {
        "lat": 21.14,
        "lon": 79.08,
        "expected_district": "Nagpur",
        "season": "kharif",
        "expected_crops": ["cotton", "tur", "soybean"],
        "in_maharashtra": True,
    },
    "Barshi": {
        "lat": 18.23,
        "lon": 75.69,
        "expected_district": "Solapur",
        "season": "rabi",
        "expected_crops": ["jowar", "tur"],
        "in_maharashtra": True,
    },
    "Khopoli": {
        "lat": 18.78,
        "lon": 73.34,
        "expected_district": "Raigad",
        "season": "kharif",
        "expected_crops": ["rice", "sugarcane", "no_crop"],
        "in_maharashtra": True,
    },
    "Wadhode": {
        "lat": 21.15,
        "lon": 75.75,
        "expected_district": "Jalgaon",
        "season": "kharif",
        "expected_crops": ["cotton", "jowar", "tur"],
        "in_maharashtra": True,
    },
    "Indore": {
        "lat": 22.72,
        "lon": 75.85,
        "expected_district": "outside",
        "season": "kharif",
        "expected_crops": [],
        "in_maharashtra": False,
    },
}


def test_akola_regression():
    sc = REGRESSION_SCENARIOS["Akola"]
    poly = point_radius_to_geojson(sc["lat"], sc["lon"], radius_km=2.0)
    ok, msg = validate_polygon(poly)
    assert ok is True
    district = locate_district(poly)
    assert "Akola" in district
    crops = plausible_crops(district, sc["season"])
    assert any(c in crops for c in ["cotton", "soybean"])


def test_amravati_regression():
    sc = REGRESSION_SCENARIOS["Amravati"]
    poly = point_radius_to_geojson(sc["lat"], sc["lon"], radius_km=2.0)
    ok, msg = validate_polygon(poly)
    assert ok is True
    district = locate_district(poly)
    assert "Amravati" in district
    crops = plausible_crops(district, sc["season"])
    assert "soybean" in crops or "cotton" in crops


def test_nagpur_regression():
    sc = REGRESSION_SCENARIOS["Nagpur"]
    poly = point_radius_to_geojson(sc["lat"], sc["lon"], radius_km=2.0)
    ok, msg = validate_polygon(poly)
    assert ok is True
    district = locate_district(poly)
    assert "Nagpur" in district
    crops = plausible_crops(district, sc["season"])
    assert "cotton" in crops


def test_barshi_solapur_regression():
    sc = REGRESSION_SCENARIOS["Barshi"]
    poly = point_radius_to_geojson(sc["lat"], sc["lon"], radius_km=2.0)
    ok, msg = validate_polygon(poly)
    assert ok is True
    district = locate_district(poly)
    assert "Solapur" in district
    crops = plausible_crops(district, "rabi")
    assert "jowar" in crops


def test_khopoli_raigad_regression():
    sc = REGRESSION_SCENARIOS["Khopoli"]
    poly = point_radius_to_geojson(sc["lat"], sc["lon"], radius_km=2.0)
    ok, msg = validate_polygon(poly)
    assert ok is True
    district = locate_district(poly)
    assert "Raigad" in district or "Pune" in district


def test_wadhode_jalgaon_regression():
    sc = REGRESSION_SCENARIOS["Wadhode"]
    poly = point_radius_to_geojson(sc["lat"], sc["lon"], radius_km=2.0)
    ok, msg = validate_polygon(poly)
    assert ok is True
    district = locate_district(poly)
    assert "Jalgaon" in district
    crops = plausible_crops(district, sc["season"])
    assert "cotton" in crops or "jowar" in crops


def test_indore_outside_maharashtra_rejection_and_override(monkeypatch):
    sc = REGRESSION_SCENARIOS["Indore"]
    poly = point_radius_to_geojson(sc["lat"], sc["lon"], radius_km=2.0)

    # 1. By default with RESTRICT_TO_MAHARASHTRA=True, Indore is rejected
    monkeypatch.setattr(config, "RESTRICT_TO_MAHARASHTRA", True)
    from utils.geo import validate_polygon
    ok, msg = validate_polygon(poly)
    assert ok is False
    assert "outside maharashtra" in msg.lower()

    # 2. When RESTRICT_TO_MAHARASHTRA=False, Indore geometry is accepted as a valid general AOI
    monkeypatch.setattr(config, "RESTRICT_TO_MAHARASHTRA", False)
    ok_global, msg_global = validate_polygon(poly)
    assert ok_global is True
    assert "Valid AOI" in msg_global
