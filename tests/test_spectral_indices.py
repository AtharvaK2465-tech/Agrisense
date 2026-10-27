"""
Unit testing for spectral indices and raster quality controls.
Validates numerical correctness on small synthetic arrays with known analytical values,
zero-denominator handling, missing-band handling (NDMI/NDRE unavailable when bands absent),
and surface fraction computations.
"""
import pytest
import numpy as np
from agents.geo_agent import _safe_normalized_difference, _compute_indices


def test_safe_normalized_difference_exact_values():
    # Known exact inputs:
    # A = [0.8, 0.4, 0.2, 0.0, 0.5]
    # B = [0.2, 0.4, 0.6, 0.0, 0.1]
    # Expected: (A - B) / (A + B)
    # [ (0.8-0.2)/(0.8+0.2) = 0.6,
    #   (0.4-0.4)/(0.4+0.4) = 0.0,
    #   (0.2-0.6)/(0.2+0.6) = -0.5,
    #   (0.0-0.0)/(0.0+0.0) = 0.0 (zero denominator handled safely),
    #   (0.5-0.1)/(0.5+0.1) = 0.666667 ]
    band_a = np.array([0.8, 0.4, 0.2, 0.0, 0.5], dtype=np.float32)
    band_b = np.array([0.2, 0.4, 0.6, 0.0, 0.1], dtype=np.float32)

    result = _safe_normalized_difference(band_a, band_b)

    assert pytest.approx(result[0], abs=1e-5) == 0.6
    assert pytest.approx(result[1], abs=1e-5) == 0.0
    assert pytest.approx(result[2], abs=1e-5) == -0.5
    assert result[3] == 0.0  # safe zero denominator
    assert pytest.approx(result[4], abs=1e-5) == 0.666667
    assert np.all(np.isfinite(result))


def test_safe_normalized_difference_handles_nan_and_inf():
    band_a = np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float32)
    band_b = np.array([0.2, 0.3, 0.1, 0.2], dtype=np.float32)

    result = _safe_normalized_difference(band_a, band_b)
    assert np.all(np.isfinite(result))
    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_compute_indices_4band_swir_rededge_unavailable():
    """When only 4 bands (RGB + NIR) are supplied, NDMI and NDRE must be marked
    unavailable rather than fabricated from Red or other approximations."""
    h, w = 8, 8
    raster_4b = np.zeros((h, w, 4), dtype=np.float32)
    raster_4b[..., 0] = 0.1  # Red
    raster_4b[..., 1] = 0.2  # Green
    raster_4b[..., 2] = 0.1  # Blue
    raster_4b[..., 3] = 0.5  # NIR

    out = _compute_indices(raster_4b)

    # NDVI = (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6667
    assert pytest.approx(float(np.mean(out["ndvi"])), abs=1e-4) == 0.6667
    assert out["ndmi"] is None
    assert out["ndre"] is None
    assert out["indices_quality"]["ndmi"]["status"] == "unavailable"
    assert "SWIR" in out["indices_quality"]["ndmi"]["reason"]
    assert out["indices_quality"]["ndre"]["status"] == "unavailable"
    assert "Red-Edge" in out["indices_quality"]["ndre"]["reason"]


def test_compute_indices_6band_full_calculation():
    h, w = 4, 4
    raster_6b = np.zeros((h, w, 6), dtype=np.float32)
    raster_6b[..., 0] = 0.1  # Red
    raster_6b[..., 1] = 0.2  # Green
    raster_6b[..., 2] = 0.1  # Blue
    raster_6b[..., 3] = 0.5  # NIR
    raster_6b[..., 4] = 0.2  # SWIR (B11)
    raster_6b[..., 5] = 0.3  # RedEdge (B05)

    out = _compute_indices(raster_6b)

    # NDVI = (0.5 - 0.1) / (0.5 + 0.1) = 0.6667
    assert pytest.approx(float(np.mean(out["ndvi"])), abs=1e-4) == 0.6667
    # NDMI = (0.5 - 0.2) / (0.5 + 0.2) = 0.3 / 0.7 = 0.4286
    assert pytest.approx(float(np.mean(out["ndmi"])), abs=1e-4) == 0.4286
    # NDRE = (0.5 - 0.3) / (0.5 + 0.3) = 0.2 / 0.8 = 0.2500
    assert pytest.approx(float(np.mean(out["ndre"])), abs=1e-4) == 0.2500

    assert out["indices_quality"]["ndmi"]["status"] == "calculated"
    assert out["indices_quality"]["ndre"]["status"] == "calculated"


def test_surface_fractions_sum_and_categories():
    # 4 pixels:
    # Pixel 0: NDVI = 0.6 (vegetation)
    # Pixel 1: NDVI = 0.15 (bare soil / fallow)
    # Pixel 2: NDVI = -0.3 (water)
    # Pixel 3: NDVI = 0.7 (vegetation)
    h, w = 2, 2
    raster = np.zeros((h, w, 4), dtype=np.float32)
    # P0: NIR=0.8, Red=0.2 -> NDVI = 0.6
    raster[0, 0, 3] = 0.8
    raster[0, 0, 0] = 0.2
    # P1: NIR=0.23, Red=0.17 -> NDVI = 0.06 / 0.4 = 0.15
    raster[0, 1, 3] = 0.23
    raster[0, 1, 0] = 0.17
    # P2: NIR=0.07, Red=0.13 -> NDVI = -0.06 / 0.20 = -0.3
    raster[1, 0, 3] = 0.07
    raster[1, 0, 0] = 0.13
    # P3: NIR=0.85, Red=0.15 -> NDVI = 0.70
    raster[1, 1, 3] = 0.85
    raster[1, 1, 0] = 0.15

    out = _compute_indices(raster)
    fractions = out["surface_fractions"]

    assert fractions["vegetation"] == 0.5  # 2 of 4
    assert fractions["bare_surface"] == 0.25  # 1 of 4
    assert fractions["water_surface"] == 0.25  # 1 of 4
    assert fractions["unclassified"] == 0.0
    total = fractions["vegetation"] + fractions["bare_surface"] + fractions["water_surface"] + fractions["unclassified"]
    assert pytest.approx(total, abs=1e-3) == 1.0


def test_compute_indices_custom_band_mapping():
    """Verify that specifying a custom band mapping correctly picks bands without assuming default order."""
    h, w = 2, 2
    # Custom 6-band order: [NIR, SWIR, RedEdge, Red, Green, Blue]
    custom_raster = np.zeros((h, w, 6), dtype=np.float32)
    custom_raster[..., 0] = 0.6  # NIR
    custom_raster[..., 1] = 0.2  # SWIR
    custom_raster[..., 2] = 0.3  # RedEdge
    custom_raster[..., 3] = 0.1  # Red
    custom_raster[..., 4] = 0.15 # Green
    custom_raster[..., 5] = 0.1  # Blue

    custom_mapping = {
        "nir": 0,
        "swir": 1,
        "rededge": 2,
        "red": 3,
        "green": 4,
        "blue": 5,
    }
    out = _compute_indices(custom_raster, band_mapping=custom_mapping)

    # NDVI: (0.6 - 0.1) / (0.6 + 0.1) = 0.5 / 0.7 = 0.7143
    assert pytest.approx(float(np.mean(out["ndvi"])), abs=1e-4) == 0.7143
    # NDMI: (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5000
    assert pytest.approx(float(np.mean(out["ndmi"])), abs=1e-4) == 0.5000
    # NDRE: (0.6 - 0.3) / (0.6 + 0.3) = 0.3 / 0.9 = 0.3333
    assert pytest.approx(float(np.mean(out["ndre"])), abs=1e-4) == 0.3333
    assert out["indices_quality"]["ndvi"]["status"] == "calculated"
    assert out["indices_quality"]["ndmi"]["status"] == "calculated"
    assert out["indices_quality"]["ndre"]["status"] == "calculated"


def test_compute_indices_mismatched_band_mapping_returns_unavailable():
    """When a band mapping omits a band or references an out-of-bounds band index,
    the index must be set to None with status 'unavailable' and an explicit reason."""
    h, w = 2, 2
    raster = np.zeros((h, w, 4), dtype=np.float32)
    raster[..., 0] = 0.1 # Red
    raster[..., 3] = 0.6 # NIR

    # Mismatched mapping: NIR points to index 99 (out of bounds)
    mismatched_mapping = {"red": 0, "green": 1, "blue": 2, "nir": 99}
    out = _compute_indices(raster, band_mapping=mismatched_mapping)

    assert out["ndvi"] is None
    assert out["indices_quality"]["ndvi"]["status"] == "unavailable"
    assert "out of bounds" in out["indices_quality"]["ndvi"]["reason"]
    assert out["valid_pixels_pct"] == 0.0


def test_compute_indices_invalid_dimensions_raises():
    """Rasters with invalid dimensionality (< 3D or empty dimensions) must raise ValueError."""
    with pytest.raises(ValueError, match="3D array"):
        _compute_indices(np.zeros((10, 10), dtype=np.float32))

    with pytest.raises(ValueError, match="positive dimensions"):
        _compute_indices(np.zeros((0, 10, 4), dtype=np.float32))

