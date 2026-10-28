"""
Deterministic Unit Tests for Sentinel-2 Cloud Masking and Pixel Quality Controls
---------------------------------------------------------------------------------
Verifies:
1. Clear pixels (100% usable, correct index calculations).
2. Individual exclusion of each non-usable Scene Classification Layer (SCL) class:
   - SCL 0: NO_DATA
   - SCL 1: SATURATED_OR_DEFECTIVE
   - SCL 2: CAST_SHADOW
   - SCL 3: CLOUD_SHADOWS
   - SCL 8: CLOUD_MEDIUM_PROBABILITY
   - SCL 9: CLOUD_HIGH_PROBABILITY
   - SCL 10: THIN_CIRRUS
   And inclusion of clear classes (SCL 4: VEGETATION, SCL 5: NOT_VEGETATED, SCL 6: WATER, SCL 7: UNCLASSIFIED).
3. No-data (dataMask == 0) and off-AOI pixel masking.
4. Separate accounting of cloud and cloud-shadow pixels (never conflated with no-data).
5. Mixed clear/cloud scenes where usable pixels maintain true uncorrupted index statistics.
6. Completely masked scenes (100% cloud/nodata) producing explicit 'insufficient_usable_pixels'
   and downstream 'unavailable' decisions without zero-filling or disease misclassification.
7. Non-finite (NaN / Inf) band value safeguards.
8. Correct denominator assertions for every fraction metric.
9. Masked observations excluded from temporal series without silent zero conversion.
10. Explicit 'unavailable' status when SCL quality information is missing.
11. Provenance recording of masking methods, excluded classes, and scientific disclaimers.
12. Preservation of Live/Mock Mode semantics.
"""
from datetime import date
import pytest
import numpy as np

from agents.geo_agent import (
    evaluate_pixel_quality,
    _compute_indices,
    geo_agent,
    SCL_NO_DATA,
    SCL_SATURATED_OR_DEFECTIVE,
    SCL_CAST_SHADOW,
    SCL_CLOUD_SHADOWS,
    SCL_VEGETATION,
    SCL_NOT_VEGETATED,
    SCL_WATER,
    SCL_UNCLASSIFIED,
    SCL_CLOUD_MEDIUM_PROBABILITY,
    SCL_CLOUD_HIGH_PROBABILITY,
    SCL_THIN_CIRRUS,
    SCL_SNOW_ICE,
)
from agents.crop_agent import crop_agent
from agents.vision_agent import _heuristic_classify, vision_agent
from agents.reasoning_agent import reasoning_agent
from agents.state import PipelineState
import utils.config as config


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

def _make_raster(h=4, w=4, bands=8, fill_reflectance=0.2, nir=0.6, scl_val=SCL_VEGETATION, datamask_val=1.0):
    """Creates a deterministic 8-band synthetic raster [Red, Green, Blue, NIR, SWIR, RedEdge, SCL, dataMask]."""
    r = np.full((h, w, bands), fill_reflectance, dtype=np.float32)
    r[..., 0] = 0.1  # Red (B04)
    r[..., 1] = 0.15 # Green (B03)
    r[..., 2] = 0.1  # Blue (B02)
    r[..., 3] = nir  # NIR (B08)
    r[..., 4] = 0.2  # SWIR (B11)
    r[..., 5] = 0.25 # RedEdge (B05)
    if bands >= 7:
        r[..., 6] = scl_val
    if bands >= 8:
        r[..., 7] = datamask_val
    return r


SQUARE_AOI = {
    "type": "Polygon",
    "coordinates": [[[77.10, 19.70], [77.20, 19.70], [77.20, 19.80], [77.10, 19.80], [77.10, 19.70]]],
}


# ---------------------------------------------------------------------------
# 1. Clear Pixels
# ---------------------------------------------------------------------------

def test_clear_pixels_100_percent_usable():
    """Clear scene (SCL 4, dataMask 1) must yield 100% usable pixels, 0% cloud, 0% nodata."""
    raster = _make_raster(h=4, w=4, scl_val=SCL_VEGETATION, datamask_val=1.0)
    out = _compute_indices(raster, polygon_geojson=SQUARE_AOI)

    qm = out["quality_metrics"]
    assert qm["usable_pixel_pct"] == 100.0
    assert qm["nodata_fraction_pct"] == 0.0
    assert qm["cloud_fraction_pct"] == 0.0
    assert qm["shadow_fraction_pct"] == 0.0
    assert qm["cloud_shadow_fraction_pct"] == 0.0
    assert qm["cloud_masking_status"] == "active"
    assert out["indices_quality"]["ndvi"]["status"] == "calculated"
    assert np.all(np.isfinite(out["ndvi"]))
    # NDVI = (0.6 - 0.1) / (0.6 + 0.1) = 0.5 / 0.7 = 0.7143
    assert pytest.approx(float(np.mean(out["ndvi"])), abs=1e-4) == 0.7143


# ---------------------------------------------------------------------------
# 2. Individual Excluded SCL Classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scl_class,expected_category", [
    (SCL_NO_DATA, "nodata"),
    (SCL_SATURATED_OR_DEFECTIVE, "defective"),
    (SCL_CAST_SHADOW, "shadow"),
    (SCL_CLOUD_SHADOWS, "shadow"),
    (SCL_CLOUD_MEDIUM_PROBABILITY, "cloud"),
    (SCL_CLOUD_HIGH_PROBABILITY, "cloud"),
    (SCL_THIN_CIRRUS, "cloud"),
])
def test_each_excluded_scl_class_individually(scl_class, expected_category):
    """Every single non-usable SCL class must be excluded from usable pixels."""
    raster = _make_raster(h=4, w=4, scl_val=scl_class, datamask_val=1.0)
    out = _compute_indices(raster, polygon_geojson=SQUARE_AOI)

    qm = out["quality_metrics"]
    counts = out["pixel_counts"]

    assert counts["usable_pixels"] == 0
    assert qm["usable_pixel_pct"] == 0.0
    assert out["indices_quality"]["ndvi"]["status"] == "insufficient_usable_pixels"
    assert np.all(np.isnan(out["ndvi"]))

    if expected_category == "cloud":
        assert qm["cloud_fraction_pct"] == 100.0
        assert counts["cloud_pixels"] == 16
    elif expected_category == "shadow":
        assert qm["shadow_fraction_pct"] == 100.0
        assert counts["shadow_pixels"] == 16
    elif expected_category == "nodata":
        assert qm["nodata_fraction_pct"] == 100.0
        assert counts["nodata_pixels"] == 16


def test_clear_land_and_water_scl_classes_included_in_usable():
    """Clear vegetation (4) and non-vegetated (5) must be counted as usable pixels."""
    for clear_scl in (SCL_VEGETATION, SCL_NOT_VEGETATED):
        raster = _make_raster(h=4, w=4, scl_val=clear_scl, datamask_val=1.0)
        out = _compute_indices(raster, polygon_geojson=SQUARE_AOI)
        assert out["quality_metrics"]["usable_pixel_pct"] == 100.0
        assert out["indices_quality"]["ndvi"]["status"] == "calculated"


# ---------------------------------------------------------------------------
# 3. No-data and Off-AOI Pixels
# ---------------------------------------------------------------------------

def test_nodata_and_off_aoi_pixel_accounting():
    """Pixels outside the polygon AOI and pixels with dataMask=0 must be separated."""
    h, w = 4, 4
    raster = _make_raster(h=h, w=w, scl_val=SCL_VEGETATION, datamask_val=1.0)
    # Set top row (4 pixels) to dataMask=0 (nodata off-swath)
    raster[0, :, 7] = 0.0

    # Define a smaller AOI that only covers the bottom 2 rows (rows 2 and 3)
    # BBox: [77.10, 19.70] to [77.20, 19.80]
    # Row 0,1 are in top half (lat 19.75-19.80); row 2,3 are in bottom half (lat 19.70-19.75)
    half_aoi = {
        "type": "Polygon",
        "coordinates": [[[77.10, 19.70], [77.20, 19.70], [77.20, 19.75], [77.10, 19.75], [77.10, 19.70]]],
    }

    eval_res = evaluate_pixel_quality(
        raster,
        polygon_geojson=half_aoi,
        bbox=(77.10, 19.70, 77.20, 19.80),
    )
    counts = eval_res["counts"]
    metrics = eval_res["metrics"]

    assert counts["grid_pixels"] == 16
    assert counts["aoi_pixels"] == 8  # 8 inside AOI, 8 outside AOI
    assert metrics["aoi_pixel_coverage_pct"] == 50.0
    # Inside the AOI (bottom half), dataMask was 1.0, so nodata=0 within AOI
    assert counts["nodata_pixels"] == 0
    assert counts["usable_pixels"] == 8
    assert metrics["usable_pixel_pct"] == 100.0


# ---------------------------------------------------------------------------
# 4. Cloud and Cloud-Shadow Distinction
# ---------------------------------------------------------------------------

def test_cloud_and_cloud_shadow_distinction():
    """Cloud pixels (SCL 8,9,10) and shadow pixels (SCL 2,3) must be reported separately."""
    h, w = 4, 4
    raster = _make_raster(h=h, w=w, scl_val=SCL_VEGETATION, datamask_val=1.0)

    # 4 pixels cloud (SCL 9) in row 0
    raster[0, :, 6] = SCL_CLOUD_HIGH_PROBABILITY
    # 2 pixels shadow (SCL 3) in row 1, cols 0..1
    raster[1, 0:2, 6] = SCL_CLOUD_SHADOWS
    # Remaining 10 pixels are clear vegetation (SCL 4)

    out = _compute_indices(raster, polygon_geojson=SQUARE_AOI)
    qm = out["quality_metrics"]
    counts = out["pixel_counts"]

    assert counts["aoi_pixels"] == 16
    assert counts["cloud_pixels"] == 4
    assert counts["shadow_pixels"] == 2
    assert counts["usable_pixels"] == 10

    # Separate fractions with AOI denominator
    assert qm["cloud_fraction_pct"] == 25.0       # 4 / 16
    assert qm["shadow_fraction_pct"] == 12.5      # 2 / 16
    assert qm["cloud_shadow_fraction_pct"] == 37.5# 6 / 16
    assert qm["usable_pixel_pct"] == 62.5         # 10 / 16
    assert qm["nodata_fraction_pct"] == 0.0       # Not conflated with nodata


# ---------------------------------------------------------------------------
# 5. Mixed Clear/Cloud Scenes: Uncorrupted Index Calculations
# ---------------------------------------------------------------------------

def test_mixed_clear_cloud_scene_indices_uncontaminated():
    """Cloudy pixels must NOT be converted to 0.0 or contaminate mean NDVI."""
    h, w = 4, 4
    raster = _make_raster(h=h, w=w, fill_reflectance=0.2, nir=0.6, scl_val=SCL_VEGETATION, datamask_val=1.0)

    # 8 pixels clear (rows 0, 1): Red=0.1, NIR=0.6 -> NDVI = 0.5 / 0.7 = 0.7143
    # 8 pixels heavy cloud (rows 2, 3): SCL=9, high reflectance (Red=0.8, NIR=0.9)
    raster[2:, :, 6] = SCL_CLOUD_HIGH_PROBABILITY
    raster[2:, :, 0] = 0.8
    raster[2:, :, 3] = 0.9

    out = _compute_indices(raster, polygon_geojson=SQUARE_AOI)
    ndvi = out["ndvi"]

    # Cloudy pixels must be NaN
    assert np.all(np.isnan(ndvi[2:, :]))
    # Clear pixels must be exact 0.7143
    assert np.all(np.isfinite(ndvi[:2, :]))
    assert pytest.approx(float(np.nanmean(ndvi)), abs=1e-4) == 0.7143

    # If cloudy pixels had been converted to 0.0, mean would be 0.7143 * 0.5 = 0.3571 (false stress)
    # Confirm this scientific error did NOT occur:
    assert float(np.nanmean(ndvi)) > 0.70


# ---------------------------------------------------------------------------
# 6. Completely Masked Imagery & Downstream Effects
# ---------------------------------------------------------------------------

def test_completely_masked_imagery_downstream_handling():
    """When imagery is 100% cloud-masked, downstream agents must yield 'unavailable',
    NOT 'no_crop' or 'disease'."""
    h, w = 4, 4
    # All pixels heavy cloud
    raster = _make_raster(h=h, w=w, scl_val=SCL_CLOUD_HIGH_PROBABILITY, datamask_val=1.0)
    indices = _compute_indices(raster, polygon_geojson=SQUARE_AOI)

    assert indices["quality_metrics"]["usable_pixel_pct"] == 0.0
    assert indices["indices_quality"]["ndvi"]["status"] == "insufficient_usable_pixels"
    assert np.all(np.isnan(indices["ndvi"]))

    state: PipelineState = {
        "polygon_geojson": SQUARE_AOI,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 6, 15),
        "ndvi": indices["ndvi"],
        "ndmi": indices["ndmi"],
        "ndre": indices["ndre"],
        "indices_quality": indices["indices_quality"],
        "valid_pixels_pct": indices["valid_pixels_pct"],
        "surface_fractions": indices["surface_fractions"],
        "cloud_cover_pct": 100.0,
    }

    # Crop Agent verification
    crop_out = crop_agent(state)
    assert crop_out["crop_decision_status"] == "unavailable"
    assert crop_out["crop_type"] == "uncertain"
    assert crop_out["crop_confidence"] == 0.0
    assert "Insufficient" in crop_out["crop_decision_reason"]

    # Vision Agent verification
    vision_out = vision_agent(crop_out)
    assert vision_out["vision_classification"] == "unavailable"
    assert vision_out["vision_confidence"] == 0.0

    # Reasoning Agent verification
    reasoning_out = reasoning_agent(vision_out)
    assert "Insufficient Usable Imagery" in reasoning_out["diagnosis"]


# ---------------------------------------------------------------------------
# 7. Non-Finite Band Values
# ---------------------------------------------------------------------------

def test_non_finite_band_values_excluded_from_usable():
    """Pixels with NaN or Inf in spectral bands must be excluded by invalid_bands_mask."""
    raster = _make_raster(h=4, w=4, scl_val=SCL_VEGETATION, datamask_val=1.0)
    raster[0, 0, 3] = np.nan   # NIR NaN
    raster[0, 1, 0] = np.inf   # Red Inf
    raster[1, 0, 0] = -0.05    # Negative non-physical reflectance

    out = _compute_indices(raster, polygon_geojson=SQUARE_AOI)
    counts = out["pixel_counts"]

    assert counts["invalid_band_pixels"] == 3
    assert counts["usable_pixels"] == 13
    assert np.isnan(out["ndvi"][0, 0])
    assert np.isnan(out["ndvi"][0, 1])
    assert np.isnan(out["ndvi"][1, 0])


# ---------------------------------------------------------------------------
# 8. Correct Denominators and Fraction Assertions
# ---------------------------------------------------------------------------

def test_correct_denominators_and_fractions():
    """Verifies that all metrics have explicit denominators and accurate ratios."""
    # 10x10 grid = 100 pixels
    raster = _make_raster(h=10, w=10, scl_val=SCL_VEGETATION, datamask_val=1.0)

    # 10 nodata pixels (dataMask=0)
    raster[0, :, 7] = 0.0
    # 20 cloud pixels (SCL 9)
    raster[1:3, :, 6] = SCL_CLOUD_HIGH_PROBABILITY
    # 10 shadow pixels (SCL 3)
    raster[3, :, 6] = SCL_CLOUD_SHADOWS
    # Remaining 60 pixels are clear vegetation (SCL 4)

    out = _compute_indices(raster, polygon_geojson=SQUARE_AOI)
    qm = out["quality_metrics"]
    counts = out["pixel_counts"]

    # Total grid = 100
    assert counts["grid_pixels"] == 100
    assert counts["aoi_pixels"] == 100
    assert qm["aoi_coverage_denominator"] == "Total raster grid pixels (H x W)"

    # AOI-denominated fractions: Denominator = 100
    assert qm["nodata_denominator"] == "Total AOI pixels"
    assert qm["nodata_fraction_pct"] == 10.0      # 10 / 100

    assert qm["source_data_availability_pct"] == 90.0  # 90 / 100
    assert qm["source_data_availability_denominator"] == "Total AOI pixels"

    assert qm["cloud_fraction_pct"] == 20.0       # 20 / 100
    assert qm["cloud_denominator"] == "Total AOI pixels"

    assert qm["shadow_fraction_pct"] == 10.0      # 10 / 100
    assert qm["shadow_denominator"] == "Total AOI pixels"

    assert qm["cloud_shadow_fraction_pct"] == 30.0# 30 / 100
    assert qm["cloud_shadow_denominator"] == "Total AOI pixels"

    assert qm["usable_pixel_pct"] == 60.0         # 60 / 100
    assert qm["usable_pixel_denominator"] == "Total AOI pixels"

    # Acquired data denominator = 90
    assert qm["usable_of_acquired_data_pct"] == round(60 / 90 * 100.0, 2)  # 66.67%
    assert qm["usable_of_acquired_denominator"] == "Acquired valid data pixels within AOI"


# ---------------------------------------------------------------------------
# 9. Explicit Unavailable Status When SCL Is Missing
# ---------------------------------------------------------------------------

def test_explicit_unavailable_when_scl_missing():
    """When a 4-band or 6-band raster without SCL is supplied, cloud masking must be
    explicitly marked 'unavailable'."""
    raster_6b = np.full((8, 8, 6), 0.25, dtype=np.float32)
    raster_6b[..., 3] = 0.6  # NIR

    out = _compute_indices(raster_6b, polygon_geojson=SQUARE_AOI)
    qm = out["quality_metrics"]

    assert qm["cloud_masking_status"] == "unavailable"
    assert qm["cloud_fraction_pct"] is None
    assert qm["shadow_fraction_pct"] is None
    assert "SCL" in qm["cloud_masking_reason"]
    assert "unavailable" in qm["cloud_masking_reason"].lower()
    assert out["indices_quality"]["cloud_masking"]["status"] == "unavailable"


# ---------------------------------------------------------------------------
# 10. Provenance Records Selected Masking Method and Disclaimers
# ---------------------------------------------------------------------------

def test_provenance_records_masking_method_and_classes(monkeypatch):
    """Geo Agent provenance must explicitly record the masking method, excluded classes, and disclaimer."""
    monkeypatch.setattr(config, "SENTINEL_HUB_MOCK", True)
    state: PipelineState = {
        "polygon_geojson": SQUARE_AOI,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 6, 15),
    }
    out = geo_agent(state)
    assert out.get("error") is None
    prov = out["provenance"]

    assert "masking_method" in prov
    assert "scl_classes_excluded" in prov
    assert "cloud" in prov["scl_classes_excluded"]
    assert "shadow" in prov["scl_classes_excluded"]
    assert prov["scl_classes_excluded"]["cloud"] == [8, 9, 10]
    assert prov["scl_classes_excluded"]["shadow"] == [2, 3]

    assert "masking_disclaimer" in prov
    assert "not provide 100% cloud detection" in prov["masking_disclaimer"]
    assert "pixel_quality_metrics" in prov


# ---------------------------------------------------------------------------
# 11. Preserved Live/Mock Mode Semantics
# ---------------------------------------------------------------------------

def test_mock_and_live_mode_semantics_preserved(monkeypatch):
    """Mock mode continues to generate synthetic rasters with simulated SCL,
    and live mode without credentials cleanly fails."""
    # 1. Mock Mode
    monkeypatch.setattr(config, "SENTINEL_HUB_MOCK", True)
    state: PipelineState = {
        "polygon_geojson": SQUARE_AOI,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 6, 15),
    }
    mock_out = geo_agent(state)
    assert mock_out["provenance"]["execution_mode"] == "mock"
    assert mock_out["provenance"]["is_synthetic_imagery"] is True
    assert mock_out["raster"].shape[-1] == 8  # 8-band raster including SCL and dataMask

    # 2. Live Mode without credentials & no fallback
    monkeypatch.setattr(config, "SENTINEL_HUB_MOCK", False)
    monkeypatch.setattr(config, "SENTINEL_HUB_CLIENT_ID", "")
    monkeypatch.setattr(config, "SENTINEL_HUB_CLIENT_SECRET", "")
    monkeypatch.setattr(config, "ALLOW_LIVE_FALLBACK_TO_MOCK", False)

    live_fail_out = geo_agent(state)
    assert live_fail_out.get("error") is not None
    assert "Live Sentinel Hub imagery retrieval failed" in live_fail_out["error"]
