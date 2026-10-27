"""
Deterministic Boundary Testing for Live Sentinel-2 Ingestion
-------------------------------------------------------------
Verifies the Sentinel Hub integration boundary using mocked API responses.
Tests:
1. Successful catalog search and Process API 6-band ingestion.
2. Missing credentials validation.
3. Authentication failure handling (HTTP 401/403).
4. Request timeout and network failure handling.
5. Rate limit (HTTP 429) and server error (HTTP 5xx) handling.
6. Catalog returning zero scenes in observation window.
7. Unexpected raster shape from provider.
8. Cloud-masked / 100% nodata output rejection.
9. Credential sanitization (secrets never leaked in error messages).
10. Strict separation of Live Mode, Mock Mode, and explicit fallback execution.
"""
from datetime import date
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from agents.geo_agent import _real_fetch_imagery, geo_agent
import utils.config as config


TEST_AOI = {
    "type": "Polygon",
    "coordinates": [[[77.10, 19.70], [77.15, 19.70], [77.15, 19.75], [77.10, 19.75], [77.10, 19.70]]],
}

TEST_STATE = {
    "polygon_geojson": TEST_AOI,
    "date_start": date(2026, 6, 1),
    "date_end": date(2026, 6, 15),
}


def test_sentinel_hub_missing_credentials_raises():
    """Missing credentials must cleanly raise ValueError without attempting network requests."""
    with patch("utils.config.get_sentinel_hub_credentials", return_value=("", "")):
        with pytest.raises(ValueError, match="Missing Sentinel Hub credentials"):
            _real_fetch_imagery(TEST_STATE)


def test_sentinel_hub_successful_catalog_and_process_fetch():
    """Verify end-to-end mocked Sentinel Hub flow: catalog search -> request -> 6-band raster."""
    mock_scene = {
        "id": "S2A_MSIL2A_20260610T0530",
        "properties": {
            "datetime": "2026-06-10T05:30:00Z",
            "eo:cloud_cover": 4.5,
            "grid:code": "43QEA",
        },
    }

    # Synthetic 6-band reflectance raster: (32, 32, 6)
    mock_raster = np.full((32, 32, 6), 0.20, dtype=np.float32)
    mock_raster[..., 3] = 0.55  # NIR

    mock_catalog = MagicMock()
    mock_catalog.search.return_value = [mock_scene]

    mock_request = MagicMock()
    mock_request.get_data.return_value = [mock_raster]

    with patch("utils.config.get_sentinel_hub_credentials", return_value=("mock-id-123", "mock-secret-456")), \
         patch("sentinelhub.SentinelHubCatalog", return_value=mock_catalog), \
         patch("sentinelhub.SentinelHubRequest", return_value=mock_request):
        
        raster, cloud_cover, meta = _real_fetch_imagery(TEST_STATE)

        assert raster.shape == (32, 32, 6)
        assert cloud_cover == 4.5
        assert meta["scene_id"] == "S2A_MSIL2A_20260610T0530"
        assert meta["is_synthetic"] is False
        assert "Process API" in meta["provider"]
        assert meta["resolution_m"] >= 10.0


def test_sentinel_hub_authentication_failure_401_403():
    """HTTP 401/403 or invalid credentials must raise PermissionError without leaking secrets."""
    client_id = "real-id-never-leak"
    client_sec = "real-secret-never-leak"

    mock_catalog = MagicMock()
    mock_catalog.search.side_effect = Exception(f"Failed to download from https://services.sentinel-hub.com with 401 Unauthorized for {client_id}")

    with patch("utils.config.get_sentinel_hub_credentials", return_value=(client_id, client_sec)), \
         patch("sentinelhub.SentinelHubCatalog", return_value=mock_catalog):

        with pytest.raises(PermissionError) as exc_info:
            _real_fetch_imagery(TEST_STATE)

        err_msg = str(exc_info.value)
        assert "authentication failed" in err_msg.lower()
        # Confirm credentials were never leaked into exception string
        assert client_id not in err_msg
        assert client_sec not in err_msg


def test_sentinel_hub_rate_limit_429():
    """HTTP 429 rate limit must raise RuntimeError with backoff recommendation."""
    mock_catalog = MagicMock()
    mock_catalog.search.side_effect = Exception("HTTP 429 Too Many Requests: RateLimit exceeded")

    with patch("utils.config.get_sentinel_hub_credentials", return_value=("test-id", "test-sec")), \
         patch("sentinelhub.SentinelHubCatalog", return_value=mock_catalog):

        with pytest.raises(RuntimeError, match="rate limit reached"):
            _real_fetch_imagery(TEST_STATE)


def test_sentinel_hub_server_error_500():
    """HTTP 5xx server errors must raise clean RuntimeError."""
    mock_catalog = MagicMock()
    mock_catalog.search.side_effect = Exception("503 Server Error: Service Unavailable")

    with patch("utils.config.get_sentinel_hub_credentials", return_value=("test-id", "test-sec")), \
         patch("sentinelhub.SentinelHubCatalog", return_value=mock_catalog):

        with pytest.raises(RuntimeError, match="service temporarily unavailable"):
            _real_fetch_imagery(TEST_STATE)


def test_sentinel_hub_request_timeout():
    """Network connection timeout must raise TimeoutError."""
    mock_catalog = MagicMock()
    mock_catalog.search.side_effect = Exception("HTTPSConnectionPool: Read timed out after 30 seconds")

    with patch("utils.config.get_sentinel_hub_credentials", return_value=("test-id", "test-sec")), \
         patch("sentinelhub.SentinelHubCatalog", return_value=mock_catalog):

        with pytest.raises(TimeoutError, match="request timed out"):
            _real_fetch_imagery(TEST_STATE)


def test_sentinel_hub_no_scenes_found_in_interval():
    """When catalog returns 0 qualifying scenes, raises RuntimeError advising date expansion."""
    mock_catalog = MagicMock()
    mock_catalog.search.return_value = []

    with patch("utils.config.get_sentinel_hub_credentials", return_value=("test-id", "test-sec")), \
         patch("sentinelhub.SentinelHubCatalog", return_value=mock_catalog):

        with pytest.raises(RuntimeError, match="No qualifying Sentinel-2 scenes found"):
            _real_fetch_imagery(TEST_STATE)


def test_sentinel_hub_unexpected_raster_shape():
    """If provider returns unexpected band shape (< 6 bands), raises ValueError."""
    mock_scene = {"id": "S2A", "properties": {"datetime": "2026-06-10", "eo:cloud_cover": 2.0}}
    mock_catalog = MagicMock()
    mock_catalog.search.return_value = [mock_scene]

    # Only 3 bands returned instead of 6
    mock_raster_3b = np.full((32, 32, 3), 0.20, dtype=np.float32)
    mock_request = MagicMock()
    mock_request.get_data.return_value = [mock_raster_3b]

    with patch("utils.config.get_sentinel_hub_credentials", return_value=("test-id", "test-sec")), \
         patch("sentinelhub.SentinelHubCatalog", return_value=mock_catalog), \
         patch("sentinelhub.SentinelHubRequest", return_value=mock_request):

        with pytest.raises(ValueError, match="unexpected raster shape"):
            _real_fetch_imagery(TEST_STATE)


def test_sentinel_hub_all_nodata_masked_pixels():
    """If provider returns an entirely masked/nodata raster, raises RuntimeError."""
    mock_scene = {"id": "S2A", "properties": {"datetime": "2026-06-10", "eo:cloud_cover": 10.0}}
    mock_catalog = MagicMock()
    mock_catalog.search.return_value = [mock_scene]

    # All NaN raster
    mock_raster_nan = np.full((32, 32, 6), np.nan, dtype=np.float32)
    mock_request = MagicMock()
    mock_request.get_data.return_value = [mock_raster_nan]

    with patch("utils.config.get_sentinel_hub_credentials", return_value=("test-id", "test-sec")), \
         patch("sentinelhub.SentinelHubCatalog", return_value=mock_catalog), \
         patch("sentinelhub.SentinelHubRequest", return_value=mock_request):

        with pytest.raises(RuntimeError, match="insufficient valid data"):
            _real_fetch_imagery(TEST_STATE)


def test_geo_agent_live_mode_success_provenance():
    """When Live Mode retrieval succeeds, provenance unambiguously records live Sentinel-2 metadata."""
    mock_raster = np.full((32, 32, 6), 0.20, dtype=np.float32)
    mock_raster[..., 3] = 0.50
    mock_meta = {
        "scene_id": "S2B_LIVE_123",
        "datetime": "2026-06-10T10:00:00Z",
        "tile": "43QEA",
        "provider": "Copernicus Sentinel-2 L2A (Sentinel Hub Process API)",
        "cloud_cover": 5.0,
        "is_synthetic": False,
        "resolution_m": 10.0,
        "resampling_note": "20m bands resampled to 10m",
    }

    with patch("utils.config.is_sentinel_hub_mock", return_value=False), \
         patch("agents.geo_agent._real_fetch_imagery", return_value=(mock_raster, 5.0, mock_meta)):

        out = geo_agent(TEST_STATE)
        assert out.get("error") is None
        prov = out["provenance"]
        assert prov["execution_mode"] == "live"
        assert prov["imagery_source"] == "sentinel-2-l2a"
        assert prov["is_synthetic_imagery"] is False
        assert prov["scene_id"] == "S2B_LIVE_123"
        assert prov["resolution_m"] == 10.0
        assert "live_fallback_to_mock" not in prov["execution_mode"]


def test_geo_agent_live_fallback_prominently_disclosed():
    """When live mode fails and fallback is permitted, provenance and fallback note clearly mark synthetic data."""
    with patch("utils.config.is_sentinel_hub_mock", return_value=False), \
         patch("utils.config.ALLOW_LIVE_FALLBACK_TO_MOCK", True), \
         patch("agents.geo_agent._real_fetch_imagery", side_effect=Exception("Connection timed out")):

        out = geo_agent(TEST_STATE)
        assert out.get("error") is None
        prov = out["provenance"]
        assert prov["execution_mode"] == "live_fallback_to_mock"
        assert prov["imagery_source"] == "synthetic"
        assert prov["is_synthetic_imagery"] is True
        assert "NOTICE: Live Sentinel Hub retrieval failed" in out["geo_fallback_note"]


def test_geo_agent_live_failure_without_fallback_yields_hard_error():
    """When ALLOW_LIVE_FALLBACK_TO_MOCK=False, failure in live fetch immediately stops with hard error."""
    with patch("utils.config.is_sentinel_hub_mock", return_value=False), \
         patch("utils.config.ALLOW_LIVE_FALLBACK_TO_MOCK", False), \
         patch("agents.geo_agent._real_fetch_imagery", side_effect=Exception("Auth failed")):

        out = geo_agent(TEST_STATE)
        assert out.get("error") is not None
        assert "Live Sentinel Hub imagery retrieval failed" in out["error"]
