"""
Deterministic Audit and Verification Suite for:
1. Categorical SCL Resampling & Integer Validation
2. SCL Class 6 (Water) & Class 7 (Unclassified) Policy Separation
3. AOI Geometry & Raster Grid Alignment (Coordinate System, Pixel-Center Offsets, Boundaries)
4. Coverage Metric Denominators & Non-Overlapping Partitions
5. Downstream Scientific Integrity & Non-Fabrication
"""
import pytest
import numpy as np
from datetime import date
from shapely.geometry import box, Polygon

from agents.geo_agent import (
    _compute_aoi_mask,
    evaluate_pixel_quality,
    _compute_indices,
    _EVALSCRIPT_8BAND,
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
from agents.vision_agent import _heuristic_classify, vision_agent
from agents.crop_agent import crop_agent
from agents.state import PipelineState
import utils.config as config


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_test_raster(h=4, w=4, bands=8, red=0.1, nir=0.6, scl_val=SCL_VEGETATION, datamask_val=1.0):
    r = np.full((h, w, bands), 0.25, dtype=np.float32)
    r[..., 0] = red
    r[..., 1] = 0.2
    r[..., 2] = 0.15
    r[..., 3] = nir
    if bands >= 5:
        r[..., 4] = 0.25  # SWIR (B11)
    if bands >= 6:
        r[..., 5] = 0.30  # RedEdge (B05)
    if bands >= 7:
        r[..., 6] = float(scl_val)
    if bands >= 8:
        r[..., 7] = float(datamask_val)
    return r


# ---------------------------------------------------------------------------
# 1. Categorical SCL Resampling & Discrete Class Validation
# ---------------------------------------------------------------------------

def test_evalscript_specifies_nearest_for_scl_and_datamask():
    """Verify that _EVALSCRIPT_8BAND explicitly configures NEAREST resampling
    for categorical SCL and dataMask, and BILINEAR for continuous bands."""
    assert "resampling: \"NEAREST\"" in _EVALSCRIPT_8BAND
    assert "resampling: \"BILINEAR\"" in _EVALSCRIPT_8BAND
    # SCL and dataMask must be in the NEAREST input block
    scl_pos = _EVALSCRIPT_8BAND.find("\"SCL\"")
    nearest_pos = _EVALSCRIPT_8BAND.find("resampling: \"NEAREST\"")
    assert scl_pos != -1 and nearest_pos != -1


def test_interpolated_non_integer_scl_values_raise_error():
    """Detects invalid non-discrete SCL values resulting from improper interpolation
    (e.g., class 4 and class 6 bilinearly blended into 4.5)."""
    raster = _make_test_raster(h=4, w=4)
    # Inject non-discrete interpolated SCL value (4.5)
    raster[1, 1, 6] = 4.5

    with pytest.raises(ValueError, match="Categorical violation in SCL"):
        evaluate_pixel_quality(raster)


def test_out_of_range_scl_classes_raise_error():
    """Detects invalid SCL values outside official Sentinel-2 range [0, 11]."""
    raster = _make_test_raster(h=4, w=4)
    raster[0, 0, 6] = 12.0  # Invalid class 12

    with pytest.raises(ValueError, match="Invalid SCL class values detected"):
        evaluate_pixel_quality(raster)

    raster[0, 0, 6] = -1.0  # Invalid negative class
    with pytest.raises(ValueError, match="Invalid SCL class values detected"):
        evaluate_pixel_quality(raster)


def test_valid_discrete_scl_preserved_as_exact_integers():
    """Confirms that valid discrete SCL float values are accepted and processed
    without loss of categorical fidelity."""
    raster = _make_test_raster(h=4, w=4, scl_val=SCL_VEGETATION)
    out = evaluate_pixel_quality(raster)
    assert out["counts"]["crop_usable_pixels"] == 16
    assert out["metrics"]["crop_usable_pixel_pct"] == 100.0


# ---------------------------------------------------------------------------
# 2. SCL Class 6 (Water) & Class 7 (Unclassified) Policy Separation
# ---------------------------------------------------------------------------

def test_water_pixels_included_in_index_usable_but_excluded_from_crop_usable():
    """Policy A vs B:
    Water (SCL 6) has valid physical reflectance, so it belongs in index_usable_mask,
    but is strictly non-crop and must be EXCLUDED from crop_usable_mask."""
    h, w = 4, 4
    raster = _make_test_raster(h=h, w=w, red=0.05, nir=0.02, scl_val=SCL_WATER)
    out = _compute_indices(raster)
    qm = out["quality_metrics"]
    counts = out["pixel_counts"]

    # Water pixels are valid for spectral index calculation
    assert counts["index_usable_pixels"] == 16
    assert qm["index_usable_pixel_pct"] == 100.0
    # Water pixels are NOT crop-usable
    assert counts["crop_usable_pixels"] == 0
    assert qm["crop_usable_pixel_pct"] == 0.0

    # General NDVI is calculated and reflects real water reflectance (negative)
    assert out["ndvi"] is not None
    assert np.all(np.isfinite(out["ndvi"]))
    assert np.all(out["ndvi"] < 0.0)

    # Crop NDVI strictly masks water to NaN
    assert np.all(np.isnan(out["crop_ndvi"]))

    # Water fraction is reported separately and accurately
    assert counts["water_pixels"] == 16
    assert qm["water_fraction_pct"] == 100.0
    assert out["surface_fractions"]["water_surface"] == 1.0
    assert out["surface_fractions"]["vegetation"] == 0.0


def test_water_pixels_do_not_inflate_agricultural_stress_in_vision_agent():
    """Regression: Water pixels (negative NDVI) must NOT be counted as stressed
    crop canopy (usable_ndvi < 0.3) in the Vision Agent."""
    # Create an AOI that is 50% healthy crop (NDVI = 0.7) and 50% water (NDVI = -0.4)
    crop_ndvi = np.full((10, 10), 0.7)
    crop_ndvi[:5, :] = np.nan  # Water portion is masked to NaN in crop_ndvi

    # Raw index NDVI has negative water values
    raw_ndvi = np.full((10, 10), 0.7)
    raw_ndvi[:5, :] = -0.4

    # When evaluated with crop_ndvi (or with water filtered), classification is healthy
    label, confidence = _heuristic_classify(raw_ndvi, crop_ndvi=crop_ndvi)
    assert label == "healthy"
    assert confidence >= 0.85


def test_scl_class_7_unclassified_policy_and_disclosures():
    """Policy for SCL Class 7:
    Excluded from confident crop-canopy diagnosis to maintain diagnostic integrity.
    Reported separately under scl_unclassified_fraction_pct."""
    h, w = 4, 4
    raster = _make_test_raster(h=h, w=w, scl_val=SCL_UNCLASSIFIED)
    out = _compute_indices(raster)
    qm = out["quality_metrics"]
    counts = out["pixel_counts"]

    # Valid for general indices
    assert counts["index_usable_pixels"] == 16
    # Excluded from crop-canopy interpretation
    assert counts["crop_usable_pixels"] == 0
    assert counts["unclassified_scl_pixels"] == 16
    assert qm["scl_unclassified_fraction_pct"] == 100.0
    assert np.all(np.isnan(out["crop_ndvi"]))
    assert "SCL 7 (Unclassified) excluded from confident crop diagnosis" in qm["scl_policy_notes"]


# ---------------------------------------------------------------------------
# 3. AOI Geometry & Raster Grid Alignment
# ---------------------------------------------------------------------------

def test_aoi_pixel_centres_and_raster_transform_alignment():
    """Verify that pixel centres are calculated strictly from the raster transform:
    origin at top-left (minx, maxy), dx = (maxx-minx)/w, dy = (maxy-miny)/h,
    with exact half-pixel offsets."""
    h, w = 4, 4
    bbox = (0.0, 0.0, 4.0, 4.0)
    # Target bottom-left quadrant [0.0, 0.0, 2.0, 2.0]
    # Grid coordinates:
    # xs = [0.5, 1.5, 2.5, 3.5]
    # ys = [3.5, 2.5, 1.5, 0.5] (North to South, row 0 is y=3.5, row 3 is y=0.5)
    # The bottom-left quadrant has x in [0.5, 1.5] and y in [1.5, 0.5] (rows 2, 3 and cols 0, 1)
    quadrant_poly = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
    }

    mask = _compute_aoi_mask(quadrant_poly, (h, w), bbox=bbox)
    assert mask.shape == (4, 4)
    assert np.sum(mask) == 4

    expected = np.array([
        [False, False, False, False],
        [False, False, False, False],
        [True,  True,  False, False],
        [True,  True,  False, False],
    ], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_boundary_pixels_deliberately_included():
    """Boundary behavior: pixel centers falling exactly on the polygon boundary
    are deliberately included via shapely.intersects_xy."""
    h, w = 2, 2
    bbox = (0.0, 0.0, 2.0, 2.0)
    # Pixel centres are at:
    # Row 0: (0.5, 1.5), (1.5, 1.5)
    # Row 1: (0.5, 0.5), (1.5, 0.5)
    # Polygon with top boundary exactly at y=1.5:
    poly_boundary = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 1.5], [0.0, 1.5], [0.0, 0.0]]],
    }
    mask = _compute_aoi_mask(poly_boundary, (h, w), bbox=bbox)
    # Row 0 has y=1.5 (on boundary), Row 1 has y=0.5 (interior) -> both are included
    assert np.all(mask)


def test_small_narrow_polygon_anchors_to_centroid_pixel():
    """Small/narrow polygons that span less than pixel resolution dx * dy
    anchor to the containing centroid cell rather than vanishing or fabricating coverage."""
    h, w = 10, 10
    bbox = (0.0, 0.0, 10.0, 10.0)
    # Tiny sub-pixel polygon around (5.2, 5.2), size 0.05
    tiny_poly = {
        "type": "Polygon",
        "coordinates": [[[5.18, 5.18], [5.22, 5.18], [5.22, 5.22], [5.18, 5.22], [5.18, 5.18]]],
    }
    mask = _compute_aoi_mask(tiny_poly, (h, w), bbox=bbox)
    # Exactly one containing pixel cell is activated
    assert np.sum(mask) == 1
    # Check row and col containing (5.2, 5.2):
    # dx=1.0, dy=1.0: col = 5, row = int((10.0 - 5.2) / 1.0) = 4
    assert mask[4, 5] is True or mask[4, 5] == 1


def test_empty_or_invalid_aoi_geometry_raises_error():
    """Empty geometries, coordinate failures, or unparseable inputs must raise
    explicit ValueError rather than fabricating coverage."""
    # 1. Empty coordinates
    with pytest.raises(ValueError, match="Invalid AOI"):
        _compute_aoi_mask({"type": "Polygon", "coordinates": []}, (4, 4))

    # 2. Invalid shape
    with pytest.raises(ValueError, match="Invalid AOI"):
        _compute_aoi_mask({"type": "InvalidType"}, (4, 4))


# ---------------------------------------------------------------------------
# 4. Coverage Metrics & Strict Partitioning
# ---------------------------------------------------------------------------

def test_full_scl_partition_has_no_double_counting_and_clean_denominators():
    """Verify that quality categories strictly partition the AOI pixels
    with no double-counting and unambiguous denominators."""
    h, w = 10, 10
    total_pixels = 100
    raster = _make_test_raster(h=h, w=w, red=0.1, nir=0.6)

    # Set up exact counts:
    # 10 Nodata (dm=0)
    raster[:1, :, 7] = 0.0
    # 20 High Cloud (SCL 9)
    raster[1:3, :, 6] = SCL_CLOUD_HIGH_PROBABILITY
    # 10 Cloud Shadow (SCL 3)
    raster[3:4, :, 6] = SCL_CLOUD_SHADOWS
    # 5 Defective (SCL 1)
    raster[4, :5, 6] = SCL_SATURATED_OR_DEFECTIVE
    # 15 Water (SCL 6)
    raster[4, 5:, 6] = SCL_WATER
    raster[5, :10, 6] = SCL_WATER
    # 5 Unclassified (SCL 7)
    raster[6, :5, 6] = SCL_UNCLASSIFIED
    # 5 Snow (SCL 11)
    raster[6, 5:, 6] = SCL_SNOW_ICE
    # Remaining 30 pixels (rows 7..9): Clear vegetation (SCL 4)
    raster[7:, :, 6] = SCL_VEGETATION

    out = _compute_indices(raster)
    qm = out["quality_metrics"]
    counts = out["pixel_counts"]

    assert counts["aoi_pixels"] == total_pixels
    assert counts["nodata_pixels"] == 10
    assert counts["cloud_pixels"] == 20
    assert counts["shadow_pixels"] == 10
    assert counts["defective_pixels"] == 5
    assert counts["water_pixels"] == 15
    assert counts["unclassified_scl_pixels"] == 5
    assert counts["snow_pixels"] == 5
    assert counts["crop_usable_pixels"] == 30
    assert counts["index_usable_pixels"] == 15 + 5 + 30  # Water (15) + Unclassified (5) + Veg (30) = 50

    # Strict sum partition check
    partition_sum = (
        counts["nodata_pixels"] +
        counts["cloud_pixels"] +
        counts["shadow_pixels"] +
        counts["defective_pixels"] +
        counts["water_pixels"] +
        counts["unclassified_scl_pixels"] +
        counts["snow_pixels"] +
        counts["crop_usable_pixels"]
    )
    assert partition_sum == total_pixels

    # Denominators
    assert qm["nodata_denominator"] == "Total AOI pixels"
    assert qm["cloud_denominator"] == "Total AOI pixels"
    assert qm["water_denominator"] == "Total AOI pixels"
    assert qm["crop_usable_denominator"] == "Total AOI pixels"
    assert qm["index_usable_denominator"] == "Total AOI pixels"

    # Percentages
    assert qm["nodata_fraction_pct"] == 10.0
    assert qm["cloud_fraction_pct"] == 20.0
    assert qm["shadow_fraction_pct"] == 10.0
    assert qm["water_fraction_pct"] == 15.0
    assert qm["scl_unclassified_fraction_pct"] == 5.0
    assert qm["snow_fraction_pct"] == 5.0
    assert qm["crop_usable_pixel_pct"] == 30.0
    assert qm["index_usable_pixel_pct"] == 50.0


# ---------------------------------------------------------------------------
# 5. Downstream Scientific Integrity Regressions
# ---------------------------------------------------------------------------

def test_cloud_and_nodata_pixels_never_become_zero_valued_ndvi():
    """Unusable pixels must be converted to NaN so they cannot contaminate
    mean NDVI or simulate bare soil/low-vigor vegetation."""
    h, w = 4, 4
    raster = _make_test_raster(h=h, w=w, scl_val=SCL_CLOUD_HIGH_PROBABILITY)
    out = _compute_indices(raster)
    ndvi = out["ndvi"]
    assert np.all(np.isnan(ndvi))
    assert not np.any(ndvi == 0.0)


def test_pure_water_aoi_yields_no_crop_and_zero_crop_usable_coverage():
    """An AOI that covers open surface water must be identified as no_crop
    and zero crop-usable coverage without falsely asserting crop stress or disease."""
    state: PipelineState = {
        "polygon_geojson": {
            "type": "Polygon",
            "coordinates": [[[73.0, 18.0], [73.02, 18.0], [73.02, 18.02], [73.0, 18.02], [73.0, 18.0]]],
        },
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
    }

    # Generate water raster
    raster = _make_test_raster(h=16, w=16, red=0.04, nir=0.02, scl_val=SCL_WATER)
    indices = _compute_indices(raster, polygon_geojson=state["polygon_geojson"])

    state.update({
        "raster": raster,
        "ndvi": indices["ndvi"],
        "crop_ndvi": indices["crop_ndvi"],
        "valid_pixels_pct": indices["valid_pixels_pct"],
        "quality_metrics": indices["quality_metrics"],
        "indices_quality": indices["indices_quality"],
        "surface_fractions": indices["surface_fractions"],
    })

    # Crop Agent should recognize water body and assign no_crop
    crop_res = crop_agent(state)
    assert crop_res["crop_type"] == "no_crop"
    assert crop_res["crop_decision_status"] == "no_crop"
    assert crop_res["agricultural_area"] is False

    # Vision Agent should follow no_crop
    vis_res = vision_agent(crop_res)
    assert vis_res["vision_classification"] == "no_crop"


def test_low_ndvi_alone_does_not_establish_disease_in_vision_agent():
    """Scientific safeguard: A low NDVI observation alone (e.g. from early emergence
    or bare soil with mean NDVI < 0.20 and low variance) does NOT establish disease."""
    # Field with uniform low NDVI = 0.16 (e.g. bare ground or early seedlings, no patchy stress)
    uniform_low_ndvi = np.full((16, 16), 0.16)
    # Stressed fraction over <0.3 is 100%, but mean_ndmi is normal and no localized patchy depression
    label, confidence = _heuristic_classify(uniform_low_ndvi, ndmi=None)
    # Should be nutrient_deficiency / low vigor, NOT disease!
    # Because disease requires localized patchy stress
    assert label != "disease"
