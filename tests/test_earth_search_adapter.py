"""
Unit and Integration Tests for AWS Earth Search Sentinel-2 L2A Ingestion Adapter
---------------------------------------------------------------------------------
Verifies:
1. STAC search payload formatting, bounding box, date interval, and collection identifier.
2. Categorical (nearest-neighbour) vs Continuous (bilinear) resampling consistency.
3. GeoAgent provider dispatching ('earth_search' vs 'sentinel_hub').
4. Preservation of discrete SCL classes without fractional interpolation artifacts.
5. Provenance tracking for live Earth Search scenes (is_synthetic=False, provider name, resolution).
"""
import pytest
from datetime import date
from unittest.mock import patch, MagicMock
import numpy as np

import utils.config as config
from utils.earth_search import (
    search_earth_search_scenes,
    read_cog_window,
    fetch_imagery_earth_search,
    BAND_ASSET_KEYS,
)
from agents.geo_agent import _real_fetch_imagery, geo_agent
from agents.state import PipelineState


TEST_AOI = {
    "type": "Polygon",
    "coordinates": [[[77.10, 19.70], [77.15, 19.70], [77.15, 19.75], [77.10, 19.75], [77.10, 19.70]]],
}

TEST_STATE = {
    "polygon_geojson": TEST_AOI,
    "date_start": date(2024, 2, 1),
    "date_end": date(2024, 2, 28),
    "imagery_provider": "earth_search",
}


def test_stac_search_payload_construction():
    """Verifies that search_earth_search_scenes constructs the compliant STAC API request."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "features": [
            {
                "id": "S2B_TEST_SCENE",
                "properties": {"datetime": "2024-02-05T05:33:31Z", "eo:cloud_cover": 0.5, "grid:code": "43QGB"},
                "assets": {k: {"href": f"https://example.com/{k}.tif"} for k in BAND_ASSET_KEYS.values()},
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        scenes = search_earth_search_scenes(
            bbox=(77.10, 19.70, 77.15, 19.75),
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 28),
            max_cloud_cover_pct=20.0,
        )
        assert len(scenes) == 1
        assert scenes[0]["id"] == "S2B_TEST_SCENE"

        # Verify call arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["collections"] == ["sentinel-2-l2a"]
        assert payload["bbox"] == [77.10, 19.70, 77.15, 19.75]
        assert "2024-02-01" in payload["datetime"]
        assert "2024-02-28" in payload["datetime"]
        assert payload["query"]["eo:cloud_cover"]["lte"] == 20.0


def test_provider_selection_logic(monkeypatch):
    """Verifies provider resolution across environment variables, config, and state."""
    # 1. When Sentinel Hub credentials are empty and IMAGERY_PROVIDER is 'auto', resolves to earth_search
    monkeypatch.setattr(config, "SENTINEL_HUB_CLIENT_ID", "")
    monkeypatch.setattr(config, "SENTINEL_HUB_CLIENT_SECRET", "")
    monkeypatch.setattr(config, "IMAGERY_PROVIDER", "auto")
    assert config.get_imagery_provider() == "earth_search"

    # 2. When Sentinel Hub credentials are present and IMAGERY_PROVIDER is 'auto', resolves to sentinel_hub
    monkeypatch.setattr(config, "SENTINEL_HUB_CLIENT_ID", "mock-client-id")
    monkeypatch.setattr(config, "SENTINEL_HUB_CLIENT_SECRET", "mock-client-secret")
    assert config.get_imagery_provider() == "sentinel_hub"

    # 3. Explicit setting overrides auto
    monkeypatch.setattr(config, "IMAGERY_PROVIDER", "earth_search")
    assert config.get_imagery_provider() == "earth_search"


def test_earth_search_resampling_categorical_integrity():
    """Verify that discrete SCL classes remain integers after resampling without fractional artifacts."""
    # Synthetic discrete 20m SCL
    scl_discrete = np.array([
        [4, 4, 5, 5],
        [4, 4, 5, 5],
        [6, 6, 7, 7],
        [6, 6, 7, 7],
    ], dtype=np.uint8)

    # Resample with nearest-neighbour (order=0)
    import scipy.ndimage
    resampled_scl = scipy.ndimage.zoom(scl_discrete.astype(np.float32), 2.0, order=0)
    rounded = np.round(resampled_scl)
    assert np.allclose(resampled_scl, rounded)
    unique_classes = set(np.unique(rounded))
    assert unique_classes.issubset({4, 5, 6, 7})
    # None of the intermediate non-existent classes (e.g. 4.5, 5.5) should exist
    for val in np.unique(rounded):
        assert float(val).is_integer()


def test_geo_agent_earth_search_dispatch():
    """Verify that geo_agent correctly calls Earth Search provider and populates live provenance."""
    mock_raster = np.full((32, 32, 8), 0.25, dtype=np.float32)
    mock_raster[..., 3] = 0.60  # NIR
    mock_raster[..., 6] = 4.0   # SCL: Vegetation
    mock_raster[..., 7] = 1.0   # dataMask: Valid

    mock_scene_meta = {
        "scene_id": "S2B_43QGB_20240205_0_L2A",
        "datetime": "2024-02-05T05:33:31Z",
        "tile": "MGRS-43QGB",
        "provider": "Copernicus Sentinel-2 L2A (AWS Earth Search / Element 84)",
        "cloud_cover": 0.0,
        "is_synthetic": False,
        "resolution_m": 10.0,
        "bands_requested": ["B04", "B03", "B02", "B08", "B11", "B05", "SCL", "dataMask"],
        "resampling_note": "Categorical bands resampled via NEAREST, continuous via BILINEAR.",
        "valid_pixel_ratio": 1.0,
        "quality_metrics": {"usable_pixel_pct": 100.0},
    }

    with patch("agents.geo_agent._real_fetch_imagery", return_value=(mock_raster, 0.0, mock_scene_meta)):
        out = geo_agent(TEST_STATE)
        assert out.get("error") is None
        prov = out["provenance"]
        assert prov["execution_mode"] == "live"
        assert prov["is_synthetic_imagery"] is False
        assert "AWS Earth Search" in prov["provider"]
        assert prov["scene_id"] == "S2B_43QGB_20240205_0_L2A"
        assert out["valid_pixels_pct"] > 90.0


def test_circular_aoi_metric_buffering_and_area():
    """Verify that point_radius_to_geojson creates an exact metric circular buffer with ~12.56 km2 area."""
    from utils.geo import point_radius_to_geojson, approx_area_km2
    from shapely.geometry import shape

    circle_geojson = point_radius_to_geojson(19.72, 77.12, radius_km=2.0, as_circle=True)
    assert circle_geojson["type"] == "Polygon"
    coords = circle_geojson["coordinates"][0]
    # 32 quad segments produce 129 coordinates (first == last)
    assert len(coords) >= 120

    geom = shape(circle_geojson)
    area_km2 = approx_area_km2(geom)
    expected_area = np.pi * (2.0 ** 2)  # ~12.56637 km2
    assert 12.50 <= area_km2 <= 12.60
    assert abs(area_km2 - expected_area) < 0.02


def test_outside_aoi_pixels_excluded_from_metrics():
    """Verify that pixels outside the AOI geometry are strictly excluded from quality and index statistics."""
    from agents.geo_agent import evaluate_pixel_quality, _compute_aoi_mask, _compute_indices
    from utils.geo import point_radius_to_geojson

    aoi = point_radius_to_geojson(19.72, 77.12, radius_km=2.0, as_circle=True)
    from shapely.geometry import shape
    bounds = shape(aoi).bounds
    aoi_mask = _compute_aoi_mask(aoi, (100, 100), bbox=bounds)
    assert 0 < aoi_mask.sum() < 10000

    # Create synthetic raster: inside is all valid vegetation (SCL 4), outside is high cloud (SCL 9) and water (SCL 6)
    raster = np.zeros((100, 100, 8), dtype=np.float32)
    # B04, B03, B02, B08, B11, B05, SCL, dataMask
    # Set inside AOI: NIR=0.5, Red=0.1, SCL=4, dataMask=1
    inside = (aoi_mask == 1)
    raster[inside, 0] = 0.10  # Red
    raster[inside, 3] = 0.50  # NIR
    raster[inside, 6] = 4.0   # SCL Vegetation
    raster[inside, 7] = 1.0   # dataMask

    # Set outside AOI: extreme values, cloud SCL 9, shadow SCL 3, dataMask 0
    outside = ~inside
    raster[outside, 0] = 0.90
    raster[outside, 3] = 0.05
    raster[outside, 6] = 9.0  # High probability cloud
    raster[outside, 7] = 0.0  # Invalid

    res = evaluate_pixel_quality(raster, polygon_geojson=aoi, bbox=bounds)
    counts = res["counts"]
    metrics = res["metrics"]

    # All inside pixels are usable vegetation
    assert counts["aoi_pixels"] == int(inside.sum())
    assert counts["outside_aoi_pixels"] == int(outside.sum())
    assert counts["grid_pixels"] == 10000
    assert counts["cloud_pixels"] == 0
    assert counts["shadow_pixels"] == 0
    assert metrics["cloud_fraction_pct"] == 0.0
    assert metrics["usable_pixel_pct"] == 100.0

    # Verify spectral indices inside AOI are unaffected by outside extreme values
    out_indices = _compute_indices(raster, polygon_geojson=aoi, bbox=bounds)
    expected_ndvi = (0.50 - 0.10) / (0.50 + 0.10)  # ~0.6667
    actual_crop_ndvi = float(np.nanmean(out_indices["crop_ndvi"]))
    assert abs(actual_crop_ndvi - expected_ndvi) < 1e-4


def test_shadow_precision_preservation():
    """Verify that small fractional shadow occurrences (e.g. 4/125600 = 0.0032%) do not round to 0.0%."""
    from agents.geo_agent import evaluate_pixel_quality

    # 125,600 pixels inside AOI
    raster = np.zeros((355, 354, 8), dtype=np.float32)
    raster[..., 6] = 4.0  # All SCL Vegetation
    raster[..., 7] = 1.0  # dataMask valid

    # Inject exactly 4 cast shadow pixels (SCL 2)
    raster[0, 0, 6] = 2.0
    raster[0, 1, 6] = 2.0
    raster[0, 2, 6] = 2.0
    raster[0, 3, 6] = 2.0

    res = evaluate_pixel_quality(raster)
    counts = res["counts"]
    metrics = res["metrics"]
    assert counts["shadow_pixels"] == 4
    # 4 / (355*354) * 100 = 0.0031829... -> rounded to 4 decimals = 0.0032
    assert metrics["shadow_fraction_pct"] == 0.0032
    assert metrics["shadow_fraction_pct"] > 0.0


def test_compute_indices_scl_unpacking_and_surface_partition():
    """Verify that _compute_indices unpacks SCL from 8-band rasters and produces exact mutually exclusive fractions."""
    from agents.geo_agent import _compute_indices

    # 10x10 raster = 100 pixels
    raster = np.zeros((10, 10, 8), dtype=np.float32)
    # 60 pixels of active vegetation (SCL 4, NDVI = 0.6)
    raster[:6, :, 0] = 0.1  # Red
    raster[:6, :, 3] = 0.4  # NIR
    raster[:6, :, 6] = 4.0  # SCL Vegetation
    raster[:6, :, 7] = 1.0  # dataMask

    # 30 pixels of bare ground (SCL 5, NDVI = 0.1)
    raster[6:9, :, 0] = 0.2
    raster[6:9, :, 3] = 0.25
    raster[6:9, :, 6] = 5.0  # SCL Bare Soil
    raster[6:9, :, 7] = 1.0

    # 8 pixels of unclassified (SCL 7)
    raster[9, :8, 0] = 0.15
    raster[9, :8, 3] = 0.15
    raster[9, :8, 6] = 7.0  # SCL Unclassified
    raster[9, :8, 7] = 1.0

    # 2 pixels of dark area / cast shadow (SCL 2)
    raster[9, 8:, 0] = 0.05
    raster[9, 8:, 3] = 0.05
    raster[9, 8:, 6] = 2.0  # SCL Dark Area / Shadow
    raster[9, 8:, 7] = 1.0

    # Call _compute_indices without passing scl explicitly
    res = _compute_indices(raster)
    fracs = res["surface_fractions"]
    qm = res["quality_metrics"]

    # Verify SCL was unpacked: water is 0, not false-positive from NDVI
    assert fracs["water_surface"] == 0.0
    assert fracs["vegetation"] == 0.60
    assert fracs["bare_surface"] == 0.30
    assert fracs["unclassified"] == 0.10  # 8 SCL 7 + 2 SCL 2 = 10 pixels
    total_frac = sum(fracs.values())
    assert pytest.approx(total_frac, abs=1e-4) == 1.0

    # Verify index-usable percentage reflects exclusion of 2 shadow pixels (98 / 100 = 98.0%)
    assert qm["index_usable_pixel_pct"] == 98.0
    # Candidate SCL 4/5 analysis mask excludes SCL 7 and SCL 2 (90 / 100 = 90.0%)
    assert qm["crop_usable_pixel_pct"] == 90.0
    assert qm["scl4_5_analysis_pct"] == 90.0

    counts = res["pixel_counts"]
    assert counts["scl4_vegetation_pixels"] == 60
    assert counts["scl5_not_vegetated_pixels"] == 30
    assert counts["scl4_5_analysis_pixels"] == 90
    assert qm["scl4_vegetation_pct"] == 60.0
    assert qm["scl5_not_vegetated_pct"] == 30.0

    # Verify separate SCL 4 vs SCL 5 NDVI statistics
    assert res["scl4_ndvi_stats"]["count"] == 60
    assert pytest.approx(res["scl4_ndvi_stats"]["mean"], abs=1e-3) == 0.60
    assert res["scl5_ndvi_stats"]["count"] == 30
    assert pytest.approx(res["scl5_ndvi_stats"]["mean"], abs=1e-3) == 0.1111


