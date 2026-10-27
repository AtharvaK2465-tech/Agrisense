import numpy as np
from agents.geo_agent import _compute_indices
from agents.vision_agent import _heuristic_classify, LABELS


def test_compute_indices_shapes():
    # 6-band raster (Red, Green, Blue, NIR, SWIR, RedEdge)
    raster_6b = np.random.uniform(0.1, 0.5, (64, 64, 6)).astype(np.float32)
    indices_6b = _compute_indices(raster_6b)
    assert indices_6b["ndvi"].shape == (64, 64)
    assert indices_6b["ndmi"].shape == (64, 64)
    assert indices_6b["ndre"].shape == (64, 64)
    assert np.all(np.isfinite(indices_6b["ndvi"]))
    assert np.all(np.isfinite(indices_6b["ndmi"]))
    assert np.all(np.isfinite(indices_6b["ndre"]))
    assert indices_6b["indices_quality"]["ndmi"]["status"] == "calculated"
    assert indices_6b["indices_quality"]["ndre"]["status"] == "calculated"

    # 4-band raster (RGB + NIR only): NDMI and NDRE must NOT be fabricated
    raster_4b = np.random.uniform(0.1, 0.5, (64, 64, 4)).astype(np.float32)
    indices_4b = _compute_indices(raster_4b)
    assert indices_4b["ndvi"].shape == (64, 64)
    assert indices_4b["ndmi"] is None
    assert indices_4b["ndre"] is None
    assert indices_4b["indices_quality"]["ndmi"]["status"] == "unavailable"
    assert indices_4b["indices_quality"]["ndre"]["status"] == "unavailable"


def test_healthy_classification_high_ndvi():
    ndvi = np.full((32, 32), 0.7)
    ndmi = np.full((32, 32), 0.4)
    label, confidence = _heuristic_classify(ndvi, ndmi)
    assert label == "healthy"
    assert label in LABELS
    assert 0 <= confidence <= 1


def test_water_stress_classification_low_ndmi():
    ndvi = np.full((32, 32), 0.15)
    ndmi = np.full((32, 32), -0.2)
    label, confidence = _heuristic_classify(ndvi, ndmi)
    assert label == "water_stress"
    assert label in LABELS


def test_no_crop_classification_non_agri():
    ndvi = np.full((32, 32), 0.08)
    ndmi = np.full((32, 32), -0.1)
    label, confidence = _heuristic_classify(ndvi, ndmi, is_non_agri=True)
    assert label == "no_crop"
    assert label in LABELS
    assert confidence >= 0.9

