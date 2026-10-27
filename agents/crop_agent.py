"""
Crop-Type Identification Agent
-------------------------------
Implements an explainable 3-stage evidence-based crop decision pipeline:

STAGE 1: Agricultural Eligibility
  Determines whether an area of interest represents genuine agricultural land
  versus non-agricultural terrain (urban built-up, residential colony, campus
  ground, industrial estate, road network, or open water body).
  SCIENTIFIC SAFEGUARD: Low NDVI alone (e.g. from early-season seedlings,
  harvested stubble, bare fallow soil, or severe drought stress) does NOT
  disqualify land as agricultural.

STAGE 2: Crop Similarity Matching
  Compares the observed 4-stage NDVI phenology series against reference
  signatures using Root-Mean-Square (RMS) spectral distance:
      RMS = sqrt(mean((observed - reference)^2))
  Smaller RMS distance represents higher morphological curve similarity.
  Inverse-distance weights are computed as explainable similarity scores
  (NOT calibrated statistical probabilities).

STAGE 3: Confidence & Uncertainty Gating
  Validates whether satellite evidence supports an identified crop or requires
  an explicit UNCERTAIN state:
    1. 'no_crop':   Non-agricultural / residential / campus / industrial / water.
    2. '<crop>':    Agricultural area with strong RMS similarity, clear margin
                    over second-best crop, and seasonal/district support.
    3. 'uncertain': Agricultural area where evidence is insufficient or where
                    two crops have near-equal similarity (margin below threshold).
"""
import numpy as np

from agents.state import PipelineState
from utils.crops import CROP_SIGNATURES, classify_ndvi_series
from utils.districts import locate_district, season_for_date, plausible_crops, detect_ground_context
from utils.config import (
    CROP_MAX_RMS_DISTANCE,
    CROP_MIN_CONFIDENCE_MARGIN,
    AGRICULTURAL_MIN_VEG_FRACTION,
)

PHENOLOGY_STAGE_LABELS = ["Early season", "Vegetative", "Peak growth", "Late season"]
CONFIDENCE_TOLERANCE = 0.15


def _seeded_rng(polygon_geojson: dict, salt: str) -> np.random.Generator:
    seed = abs(hash(str(polygon_geojson) + salt)) % (2**32)
    return np.random.default_rng(seed)


def _mock_phenology_series(polygon_geojson: dict) -> list[float]:
    """Synthesizes a 4-date NDVI time series across the growing season.
    LIVE MODE would instead average NDVI over the polygon for 3-4 real
    Sentinel-2 scenes spaced across the season (see README)."""
    rng = _seeded_rng(polygon_geojson, "phenology")
    # Only choose among valid agricultural crop signatures for synthetic test phenology
    agri_crops = [c for c in CROP_SIGNATURES.keys() if c != "no_crop"]
    true_crop = rng.choice(agri_crops)
    signature = CROP_SIGNATURES[true_crop]
    return [float(np.clip(v + rng.normal(0, 0.035), 0.0, 1.0)) for v in signature]


def _mock_baseline_ndvi(polygon_geojson: dict, current_mean_ndvi: float) -> float:
    """3-year historical average NDVI for this area at the same time of
    year, so results can be judged as 'unusual' relative to local history
    rather than a single-snapshot threshold."""
    rng = _seeded_rng(polygon_geojson, "baseline")
    years = [current_mean_ndvi + rng.normal(0, 0.05) for _ in range(3)]
    return float(np.clip(np.mean(years), 0.0, 1.0))


def crop_agent(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    polygon = state["polygon_geojson"]
    ndvi = state["ndvi"]
    current_mean_ndvi = float(np.mean(ndvi))

    district = locate_district(polygon)
    season = season_for_date(state["date_end"])

    # =========================================================================
    # STAGE 1: AGRICULTURAL ELIGIBILITY
    # =========================================================================
    # Evaluate land-use context from coordinate geofencing and OpenStreetMap
    ground_info = detect_ground_context(polygon, current_mean_ndvi)
    is_explicit_non_agri = ground_info.get("is_non_agri", False)

    # Check for surface water bodies via spectral indices:
    # Water displays negative NDVI and very low NIR reflectance.
    water_fraction = float(np.mean((ndvi < 0.0) & (ndvi > -1.0)))
    if water_fraction > 0.60:
        ground_info = {
            "is_non_agri": True,
            "category": "Water Body",
            "reason": f"Predominantly negative NDVI ({water_fraction:.0%} of pixels < 0.0) indicating surface water body.",
        }
        is_explicit_non_agri = True

    # Spatial vegetation fraction in scene (pixels displaying active photosynthetic canopy)
    veg_fraction = float(np.mean(ndvi > 0.22))

    # Determine eligibility:
    # CRITICAL PRINCIPLE: Low NDVI alone (from bare soil, recent harvest, early emergence,
    # or drought stress) does NOT make an area non-agricultural if it sits in an agricultural landscape.
    if is_explicit_non_agri:
        agricultural_area = False
        agricultural_confidence = 0.95
    else:
        agricultural_area = True
        agricultural_confidence = 0.90

    # =========================================================================
    # STAGE 2: CROP SIMILARITY MATCHING
    # =========================================================================
    if not agricultural_area:
        best_crop = "no_crop"
        crop_decision_status = "no_crop"
        confidence = 0.95
        similarity_margin = 0.0
        best_dist = 0.0
        distances = {}
        scores = {c: (0.95 if c == "no_crop" else 0.01) for c in CROP_SIGNATURES}

        series = [
            round(max(0.06, current_mean_ndvi - 0.02), 2),
            round(max(0.07, current_mean_ndvi - 0.01), 2),
            round(current_mean_ndvi, 2),
            round(current_mean_ndvi, 2),
        ]
        crop_decision_reason = (
            f"Non-agricultural terrain detected ({ground_info.get('category', 'built-up / urban / water')}). "
            f"Insufficient evidence of cultivated agricultural vegetation."
        )
    else:
        # Evaluate agricultural candidate signatures only (never compare to 'no_crop')
        agri_signatures = {c: sig for c, sig in CROP_SIGNATURES.items() if c not in ("no_crop", "uncertain")}

        series = _mock_phenology_series(polygon)
        series[-1] = current_mean_ndvi

        # Compute RMS distances and normalized inverse-distance similarity weights
        _, scores, distances = classify_ndvi_series(
            series, signatures=agri_signatures, return_distances=True
        )

        sorted_candidates = sorted(distances.items(), key=lambda kv: kv[1])
        best_cand, best_dist = sorted_candidates[0]
        second_cand, second_dist = sorted_candidates[1]

        best_score = scores[best_cand]
        second_score = scores[second_cand]
        similarity_margin = best_score - second_score

        # =========================================================================
        # STAGE 3: CONFIDENCE & UNCERTAINTY GATING
        # =========================================================================
        # 1. Absolute similarity check: does the observed series fit ANY signature?
        # 2. Margin check: is the top candidate sufficiently separated from #2?
        # 3. Contextual district/season check.
        plausible = [c for c in plausible_crops(district, season) if c not in ("no_crop", "uncertain")]

        if best_dist > CROP_MAX_RMS_DISTANCE:
            # Curve morphology does not match any known reference signature closely
            best_crop = "uncertain"
            crop_decision_status = "uncertain"
            confidence = round(best_score, 3)
            crop_decision_reason = (
                f"Agricultural area detected, but temporal NDVI pattern deviates from reference "
                f"crop signatures (RMS distance {best_dist:.3f} exceeds threshold {CROP_MAX_RMS_DISTANCE})."
            )
        elif similarity_margin < CROP_MIN_CONFIDENCE_MARGIN:
            # Top two crops are too close to distinguish reliably (e.g. cotton vs soybean ambiguity)
            best_crop = "uncertain"
            crop_decision_status = "uncertain"
            confidence = round(best_score, 3)
            crop_decision_reason = (
                f"Agricultural vegetation detected, but crop signatures for '{best_cand}' and "
                f"'{second_cand}' are ambiguous (similarity margin {similarity_margin:.1%} is "
                f"below discrimination threshold {CROP_MIN_CONFIDENCE_MARGIN:.1%})."
            )
        else:
            # Clear mathematical winner with sufficient evidence
            if plausible and best_cand not in plausible:
                # Top match isn't standard for district/season; check if a plausible candidate is close
                plausible_dists = {c: distances[c] for c in plausible if c in distances}
                if plausible_dists:
                    best_plausible = min(plausible_dists, key=plausible_dists.get)
                    p_dist = plausible_dists[best_plausible]
                    if p_dist <= CROP_MAX_RMS_DISTANCE and (p_dist - best_dist) <= 0.05:
                        best_crop = best_plausible
                        confidence = round(scores[best_plausible] * 0.90, 3)
                        crop_decision_status = "identified"
                        crop_decision_reason = (
                            f"NDVI shape closest to '{best_cand}' but re-ranked to '{best_plausible}' "
                            f"(RMS {p_dist:.3f}) to align with {district} {season} crop calendar."
                        )
                    else:
                        best_crop = "uncertain"
                        confidence = round(scores[best_cand] * 0.75, 3)
                        crop_decision_status = "uncertain"
                        crop_decision_reason = (
                            f"Satellite phenology best matches '{best_cand}', which is uncommon in "
                            f"{district} during {season} season; marked uncertain to prevent unvalidated claims."
                        )
                else:
                    best_crop = best_cand
                    confidence = round(best_score, 3)
                    crop_decision_status = "identified"
                    crop_decision_reason = f"Phenology match for '{best_cand}' (RMS distance {best_dist:.3f})."
            else:
                # Strong match with calendar agreement
                best_crop = best_cand
                confidence = round(min(best_score * 1.05, 0.97), 3)
                crop_decision_status = "identified"
                crop_decision_reason = (
                    f"Strong phenology match for '{best_cand}' (RMS distance {best_dist:.3f}, "
                    f"margin {similarity_margin:.1%}) with agricultural spatial support."
                )

    # 3-year historical baseline comparison
    is_synth_imagery = state.get("provenance", {}).get("is_synthetic_imagery", True)
    if is_synth_imagery:
        baseline = _mock_baseline_ndvi(polygon, current_mean_ndvi)
        anomaly = round(float(current_mean_ndvi - baseline), 3)
        baseline_status = "synthetic_simulation"
        baseline_reason = "Simulated 3-year baseline for demonstration. Not based on real multi-year satellite pulls."
    else:
        # In live mode without a multi-year historical retrieval pipeline
        baseline = None
        anomaly = None
        baseline_status = "unavailable"
        baseline_reason = "Multi-year historical satellite catalog retrieval was not performed for this query."

    # Spatial confidence map:
    # Expected NDVI at this stage for the decision
    if best_crop in CROP_SIGNATURES:
        expected_ndvi = CROP_SIGNATURES[best_crop][-1]
    else:
        # For 'uncertain', use mean of top candidate or general active canopy
        expected_ndvi = CROP_SIGNATURES[sorted_candidates[0][0]][-1] if agricultural_area else 0.10

    confidence_map = 1.0 - np.abs(ndvi - expected_ndvi) / CONFIDENCE_TOLERANCE
    confidence_map = np.clip(confidence_map, 0.0, 1.0)

    from utils.crops import REFERENCE_METADATA

    return {
        **state,
        "crop_type": best_crop,
        "crop_confidence": round(float(confidence), 3),
        "crop_scores": {k: round(float(v), 3) for k, v in scores.items()},
        "crop_confidence_map": confidence_map,
        "phenology_stage_labels": PHENOLOGY_STAGE_LABELS,
        "ndvi_timeseries": [round(v, 3) for v in series],
        "district": district,
        "season": season,
        "baseline_ndvi": round(baseline, 3) if baseline is not None else None,
        "baseline_status": baseline_status,
        "baseline_reason": baseline_reason,
        "ndvi_anomaly": anomaly,
        "crop_agent_notes": crop_decision_reason,
        "is_non_agri": not agricultural_area,
        "ground_context": ground_info.get("category", ""),
        "agricultural_area": agricultural_area,
        "agricultural_confidence": round(float(agricultural_confidence), 3),
        "crop_decision_status": crop_decision_status,
        "crop_decision_reason": crop_decision_reason,
        "crop_similarity_margin": round(float(similarity_margin), 3) if agricultural_area else 0.0,
        "crop_evidence": {
            "mean_ndvi": round(current_mean_ndvi, 3),
            "vegetation_fraction": round(veg_fraction, 3),
            "ground_category": ground_info.get("category", ""),
            "distances": {k: round(float(v), 3) for k, v in distances.items()} if agricultural_area else {},
            "best_rms_distance": round(float(best_dist), 3) if agricultural_area else None,
            "margin": round(float(similarity_margin), 3) if agricultural_area else None,
            "reference_metadata": REFERENCE_METADATA,
        },
    }
