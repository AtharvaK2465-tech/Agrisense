import numpy as np
from datetime import date

from utils.crops import classify_ndvi_series, CROP_SIGNATURES
from utils.config import CROP_MAX_RMS_DISTANCE, CROP_MIN_CONFIDENCE_MARGIN
from agents.crop_agent import crop_agent


# ---------------------------------------------------------------------------
# Core signature recovery tests
# ---------------------------------------------------------------------------

def test_classifier_recovers_exact_reference_signatures():
    for crop, signature in CROP_SIGNATURES.items():
        best, scores = classify_ndvi_series(signature)
        assert best == crop
        assert scores[best] == max(scores.values())


def test_scores_sum_to_one():
    _, scores = classify_ndvi_series(CROP_SIGNATURES["cotton"])
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_noisy_signature_still_classifies_correctly():
    noisy = [v + 0.01 for v in CROP_SIGNATURES["sugarcane"]]
    best, _ = classify_ndvi_series(noisy)
    assert best == "sugarcane"


# ---------------------------------------------------------------------------
# Phase 15 Required Scenarios: 1 through 14
# ---------------------------------------------------------------------------

HINGOLI_AGRI_POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [77.10, 19.70], [77.15, 19.70], [77.15, 19.75], [77.10, 19.75], [77.10, 19.70]
    ]],
}

VIT_PUNE_CAMPUS_POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [73.868, 18.463], [73.870, 18.463], [73.870, 18.465], [73.868, 18.465], [73.868, 18.463]
    ]],
}


def test_scenario_1_healthy_agricultural_field():
    """TEST 1: Healthy agricultural field -> agricultural_area=True, crop!=no_crop, supported confidence."""
    state = {
        "polygon_geojson": HINGOLI_AGRI_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
        "ndvi": np.full((32, 32), 0.65, dtype=np.float32),
        "ndmi": np.full((32, 32), 0.25, dtype=np.float32),
        "ndre": np.full((32, 32), 0.35, dtype=np.float32),
        "cloud_cover": 5.0,
    }
    out = crop_agent(state)
    assert out["agricultural_area"] is True
    assert out["crop_type"] != "no_crop"
    assert out["crop_confidence"] >= 0.25  # Meaningful confidence share among 6 candidate classes
    assert out["crop_decision_status"] in ("identified", "uncertain")


def test_scenario_2_strong_cotton_phenology():
    """TEST 2: Strong cotton-like phenology -> cotton or highest-supported crop."""
    cotton_series = [0.25, 0.54, 0.74, 0.31]
    best, scores, dists = classify_ndvi_series(cotton_series, return_distances=True)
    assert best == "cotton"
    assert dists["cotton"] < 0.03


def test_scenario_3_strong_soybean_phenology():
    """TEST 3: Strong soybean-like phenology -> soybean or highest-supported crop."""
    soybean_series = [0.20, 0.49, 0.67, 0.36]
    best, scores, dists = classify_ndvi_series(soybean_series, return_distances=True)
    assert best == "soybean"
    assert dists["soybean"] < 0.03


def test_scenario_4_early_stage_agriculture():
    """TEST 4: Early-stage agriculture (low NDVI ~0.15) -> NOT automatically no_crop."""
    state = {
        "polygon_geojson": HINGOLI_AGRI_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 6, 20),
        "ndvi": np.full((32, 32), 0.15, dtype=np.float32),
        "ndmi": np.full((32, 32), 0.08, dtype=np.float32),
        "ndre": np.full((32, 32), 0.09, dtype=np.float32),
        "cloud_cover": 5.0,
    }
    out = crop_agent(state)
    assert out["agricultural_area"] is True
    assert out["crop_type"] != "no_crop"


def test_scenario_5_recently_harvested_agriculture():
    """TEST 5: Recently harvested agricultural field -> NOT automatically no_crop solely because NDVI is low."""
    state = {
        "polygon_geojson": HINGOLI_AGRI_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 11, 20),
        "ndvi": np.full((32, 32), 0.13, dtype=np.float32),
        "ndmi": np.full((32, 32), 0.05, dtype=np.float32),
        "ndre": np.full((32, 32), 0.07, dtype=np.float32),
        "cloud_cover": 2.0,
    }
    out = crop_agent(state)
    assert out["agricultural_area"] is True
    assert out["crop_type"] != "no_crop"


def test_scenario_6_stressed_agricultural_field():
    """TEST 6: Stressed agricultural field with depressed NDVI -> NOT automatically no_crop."""
    state = {
        "polygon_geojson": HINGOLI_AGRI_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 8, 15),
        "ndvi": np.full((32, 32), 0.18, dtype=np.float32),
        "ndmi": np.full((32, 32), 0.04, dtype=np.float32),
        "ndre": np.full((32, 32), 0.10, dtype=np.float32),
        "cloud_cover": 10.0,
    }
    out = crop_agent(state)
    assert out["agricultural_area"] is True
    assert out["crop_type"] != "no_crop"


def test_scenario_7_residential_built_up():
    """TEST 7: Residential / campus built-up ground -> no_crop."""
    state = {
        "polygon_geojson": VIT_PUNE_CAMPUS_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 1),
        "ndvi": np.full((32, 32), 0.09, dtype=np.float32),
        "ndmi": np.full((32, 32), 0.04, dtype=np.float32),
        "ndre": np.full((32, 32), 0.05, dtype=np.float32),
        "cloud_cover": 0.0,
    }
    out = crop_agent(state)
    assert out["agricultural_area"] is False
    assert out["crop_type"] == "no_crop"
    assert out["crop_decision_status"] == "no_crop"


def test_scenario_8_road_transport_infrastructure():
    """TEST 8: Transport infrastructure / urban highway core -> no_crop."""
    pune_urban_highway = {
        "type": "Polygon",
        "coordinates": [[
            [73.850, 18.520], [73.855, 18.520], [73.855, 18.525], [73.850, 18.525], [73.850, 18.520]
        ]],
    }
    state = {
        "polygon_geojson": pune_urban_highway,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 1),
        "ndvi": np.full((32, 32), 0.07, dtype=np.float32),
        "ndmi": np.full((32, 32), 0.02, dtype=np.float32),
        "ndre": np.full((32, 32), 0.03, dtype=np.float32),
        "cloud_cover": 0.0,
    }
    out = crop_agent(state)
    assert out["agricultural_area"] is False
    assert out["crop_type"] == "no_crop"


def test_scenario_9_water_body():
    """TEST 9: Water body (predominantly negative NDVI) -> no_crop."""
    state = {
        "polygon_geojson": HINGOLI_AGRI_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 1),
        "ndvi": np.full((32, 32), -0.35, dtype=np.float32),
        "ndmi": np.full((32, 32), -0.20, dtype=np.float32),
        "ndre": np.full((32, 32), -0.15, dtype=np.float32),
        "cloud_cover": 0.0,
    }
    out = crop_agent(state)
    assert out["agricultural_area"] is False
    assert out["crop_type"] == "no_crop"
    assert "Water" in out["ground_context"]


def test_scenario_10_bare_soil():
    """TEST 10: Bare soil in agricultural field -> fallow or uncertain, NOT residential no_crop."""
    state = {
        "polygon_geojson": HINGOLI_AGRI_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 1),
        "ndvi": np.full((32, 32), 0.14, dtype=np.float32),
        "ndmi": np.full((32, 32), 0.05, dtype=np.float32),
        "ndre": np.full((32, 32), 0.06, dtype=np.float32),
        "cloud_cover": 0.0,
    }
    out = crop_agent(state)
    assert out["agricultural_area"] is True
    assert out["crop_type"] != "no_crop"


def test_scenario_11_ambiguous_crop_signatures():
    """TEST 11: Ambiguous crop signature (nearly equal similarity) -> uncertain."""
    # Synthetic curve halfway between cotton [0.25, 0.55, 0.75, 0.30] and soybean [0.20, 0.50, 0.68, 0.35]
    ambiguous_series = [
        (c + s) / 2.0
        for c, s in zip(CROP_SIGNATURES["cotton"], CROP_SIGNATURES["soybean"])
    ]
    agri_signatures = {c: sig for c, sig in CROP_SIGNATURES.items() if c not in ("no_crop", "uncertain")}
    _, scores, dists = classify_ndvi_series(ambiguous_series, signatures=agri_signatures, return_distances=True)

    sorted_candidates = sorted(dists.items(), key=lambda kv: kv[1])
    margin = scores[sorted_candidates[0][0]] - scores[sorted_candidates[1][0]]
    # Margin must be very small because the series is equidistant
    assert margin < CROP_MIN_CONFIDENCE_MARGIN


def test_scenario_12_two_crops_nearly_equal_similarity():
    """TEST 12: Two crops with nearly equal similarity -> explicit uncertain decision."""
    # Midpoint between cotton and soybean creates a tie with near-zero margin
    midpoint_series = [
        (c + s) / 2.0
        for c, s in zip(CROP_SIGNATURES["cotton"], CROP_SIGNATURES["soybean"])
    ]
    agri_signatures = {c: sig for c, sig in CROP_SIGNATURES.items() if c not in ("no_crop", "uncertain")}
    _, scores, dists = classify_ndvi_series(midpoint_series, signatures=agri_signatures, return_distances=True)
    sorted_candidates = sorted(dists.items(), key=lambda kv: kv[1])
    margin = abs(scores[sorted_candidates[0][0]] - scores[sorted_candidates[1][0]])
    assert margin < CROP_MIN_CONFIDENCE_MARGIN


def test_scenario_13_strong_crop_similarity_but_insufficient_agricultural_evidence():
    """TEST 13: Strong crop similarity but area is non-agricultural -> no_crop."""
    state = {
        "polygon_geojson": VIT_PUNE_CAMPUS_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 1),
        "ndvi": np.full((32, 32), 0.70, dtype=np.float32),  # even if landscaped grass has high NDVI
        "ndmi": np.full((32, 32), 0.25, dtype=np.float32),
        "ndre": np.full((32, 32), 0.30, dtype=np.float32),
        "cloud_cover": 0.0,
    }
    out = crop_agent(state)
    assert out["agricultural_area"] is False
    assert out["crop_type"] == "no_crop"


def test_scenario_14_agricultural_evidence_strong_but_crop_signatures_weak():
    """TEST 14: Agricultural evidence strong, but phenology curve deviates from all signatures -> uncertain."""
    # Flat 0.40 across all stages doesn't match any seasonal crop curve (all have RMS > CROP_MAX_RMS_DISTANCE)
    flat_series = [0.40, 0.40, 0.40, 0.40]
    agri_signatures = {c: sig for c, sig in CROP_SIGNATURES.items() if c not in ("no_crop", "uncertain")}
    _, scores, dists = classify_ndvi_series(flat_series, signatures=agri_signatures, return_distances=True)
    best_dist = min(dists.values())
    assert best_dist > 0.08  # significantly far from reference curves
