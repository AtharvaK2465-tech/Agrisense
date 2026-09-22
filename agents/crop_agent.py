"""
Crop-Type Identification Agent
-------------------------------
Classifies the drawn area's crop type using a phenology-based approach:
samples NDVI at several dates spanning the growing season and matches the
resulting curve *shape* against reference signatures for crops grown in
Maharashtra (see utils/crops.py) — real, training-data-free remote-sensing
methodology, not a black box.

The predicted crop is then cross-checked against the district's crop
calendar (utils/districts.py): a prediction implausible for the season/
district gets re-ranked and its confidence penalized, which makes the
output meaningfully more credible than an unconstrained per-pixel guess.

Also produces:
  - a per-pixel confidence map (how well each pixel's NDVI matches the
    predicted crop's expected value at this stage)
  - a 3-year seasonal NDVI baseline + anomaly, so "unusually stressed"
    claims are judged against local history rather than a bare threshold

Runs in MOCK MODE by default (synthetic multi-date NDVI, seeded by the
polygon so results are stable across re-runs). In LIVE MODE, replace
`_mock_phenology_series` with 3-4 real Sentinel Hub pulls spaced across
the season, averaged over the polygon — the classification logic below
is unchanged either way.
"""
import numpy as np

from agents.state import PipelineState
from utils.crops import CROP_SIGNATURES, classify_ndvi_series
from utils.districts import locate_district, season_for_date, plausible_crops

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
    true_crop = rng.choice(list(CROP_SIGNATURES.keys()))
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

    # Anchor the last (current) phenology point to the real fetched
    # imagery's mean NDVI so the classification and the displayed map agree.
    series = _mock_phenology_series(polygon)
    series[-1] = current_mean_ndvi

    best_crop, scores = classify_ndvi_series(series)
    confidence = scores[best_crop]

    district = locate_district(polygon)
    season = season_for_date(state["date_end"])
    plausible = plausible_crops(district, season)

    notes = [
        f"NDVI signature across {len(series)} phenology stages best matches "
        f"'{best_crop}' (confidence {confidence:.0%})."
    ]

    if plausible and best_crop not in plausible:
        plausible_scores = {c: scores[c] for c in plausible if c in scores}
        if plausible_scores:
            reweighted = max(plausible_scores, key=plausible_scores.get)
            notes.append(
                f"'{best_crop}' isn't typically grown in {district} during the "
                f"{season} season per the district crop calendar — re-ranked "
                f"to '{reweighted}', its next-best NDVI match."
            )
            best_crop = reweighted
            confidence = scores[best_crop] * 0.85  # penalize for the mismatch
    elif plausible:
        confidence = min(confidence * 1.05, 0.97)  # small boost for calendar agreement
        notes.append(f"Consistent with crops typically grown in {district} ({season} season).")

    baseline = _mock_baseline_ndvi(polygon, current_mean_ndvi)
    anomaly = current_mean_ndvi - baseline
    if anomaly < -0.08:
        notes.append(
            f"NDVI is {abs(anomaly):.2f} below this area's 3-year seasonal "
            "average — unusually stressed."
        )
    elif anomaly > 0.08:
        notes.append(
            f"NDVI is {anomaly:.2f} above this area's 3-year seasonal "
            "average — unusually vigorous."
        )
    else:
        notes.append("NDVI is within the normal range for this area/season based on the 3-year baseline.")

    expected = CROP_SIGNATURES[best_crop][-1]
    confidence_map = 1.0 - np.abs(ndvi - expected) / CONFIDENCE_TOLERANCE
    confidence_map = np.clip(confidence_map, 0.0, 1.0)

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
        "baseline_ndvi": round(baseline, 3),
        "ndvi_anomaly": round(float(anomaly), 3),
        "crop_agent_notes": " ".join(notes),
    }
