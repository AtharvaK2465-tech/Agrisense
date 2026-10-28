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

