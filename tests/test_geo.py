import pytest
from utils.geo import validate_polygon

MUMBAI_SMALL_SQUARE = {
    "type": "Polygon",
    "coordinates": [[
        [72.80, 19.00], [72.90, 19.00], [72.90, 19.10], [72.80, 19.10], [72.80, 19.00]
    ]],
}

OUTSIDE_MAHARASHTRA = {
    "type": "Polygon",
    "coordinates": [[
        [77.0, 28.0], [77.1, 28.0], [77.1, 28.1], [77.0, 28.1], [77.0, 28.0]  # Delhi
    ]],
}

HUGE_AREA = {
    "type": "Polygon",
    "coordinates": [[
        [72.7, 16.0], [80.8, 16.0], [80.8, 22.0], [72.7, 22.0], [72.7, 16.0]
    ]],
}


def test_valid_polygon_within_maharashtra():
    ok, msg = validate_polygon(MUMBAI_SMALL_SQUARE)
    assert ok is True
    assert "Valid" in msg


def test_polygon_outside_maharashtra_rejected():
    ok, msg = validate_polygon(OUTSIDE_MAHARASHTRA)
    assert ok is False
    assert "outside" in msg.lower()


def test_oversized_polygon_rejected():
    ok, msg = validate_polygon(HUGE_AREA)
    assert ok is False
    assert "exceeds" in msg.lower()


def test_malformed_geojson_rejected():
    ok, msg = validate_polygon({"type": "Polygon", "coordinates": []})
    assert ok is False


def test_wkt_polygon_validation():
    from utils.geo import validate_polygon
    wkt_poly = "POLYGON((72.8 19.0, 72.9 19.0, 72.9 19.1, 72.8 19.1, 72.8 19.0))"
    ok, msg = validate_polygon(wkt_poly)
    assert ok is True


def test_point_radius_to_geojson():
    from utils.geo import point_radius_to_geojson, validate_polygon
    geom = point_radius_to_geojson(19.0, 72.8, radius_km=2.0)
    ok, msg = validate_polygon(geom)
    assert ok is True


def test_get_aoi_metadata():
    from utils.geo import get_aoi_metadata
    meta = get_aoi_metadata(MUMBAI_SMALL_SQUARE)
    assert "area_km2" in meta
    assert "area_hectares" in meta
    assert meta["area_km2"] > 0
    assert 72.0 <= meta["centroid_lon"] <= 73.0
    assert 18.0 <= meta["centroid_lat"] <= 20.0
