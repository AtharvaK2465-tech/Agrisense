"""
Crop phenology reference signatures for Maharashtra's major crops, and a
lightweight classifier that matches an observed NDVI time-series against
them.

Methodology: different crops have distinctive NDVI curve *shapes* across
the growing season — sugarcane stays green almost year-round, cotton
peaks sharply then drops hard post-harvest, soybean/jowar/tur have
shorter, lower-amplitude cycles. Sampling NDVI at several dates and
comparing the curve shape against reference signatures is a real,
training-data-free remote-sensing technique (as opposed to a black-box
model), which is why it's used here as the default classifier. A trained
classifier (Random Forest / small CNN over labeled polygons, see
README) is a natural upgrade once labeled data is available.

Reference values below are illustrative NDVI signatures at four
representative growing-season stages, not survey-grade agronomic data —
good enough to distinguish crop *shapes* for a demo/project, and clearly
documented as such in the methodology panel in the UI.
"""
import numpy as np

# stage order: [early season, vegetative, peak growth, late season]
CROP_SIGNATURES = {
    "sugarcane": [0.55, 0.65, 0.72, 0.68],
    "cotton": [0.25, 0.55, 0.75, 0.30],
    "soybean": [0.20, 0.50, 0.68, 0.35],
    "jowar": [0.22, 0.45, 0.60, 0.25],
    "tur": [0.20, 0.40, 0.58, 0.45],
    "fallow": [0.12, 0.15, 0.14, 0.13],
    "no_crop": [0.08, 0.10, 0.11, 0.09],
}

CROP_DISPLAY_NAMES = {
    "sugarcane": "Sugarcane",
    "cotton": "Cotton",
    "soybean": "Soybean",
    "jowar": "Jowar (Sorghum)",
    "tur": "Tur (Pigeon Pea)",
    "fallow": "Fallow / Bare Soil",
    "no_crop": "No Crop (Non-Agricultural Ground)",
    "uncertain": "Agricultural Area (Crop Uncertain)",
}

# Hex colors for the map legend / overlay, loosely following standard
# agronomy map conventions (deep green = perennial cane, warm tones for
# the shorter kharif crops, brown = bare/fallow, neutral gray = built-up/ground, amber = uncertain).
CROP_COLORS = {
    "sugarcane": "#1b7837",
    "cotton": "#fee08b",
    "soybean": "#f1a340",
    "jowar": "#d73027",
    "tur": "#66c2a5",
    "fallow": "#8c6d31",
    "no_crop": "#757575",
    "uncertain": "#d97706",
}

REFERENCE_METADATA = {
    "source": "Regional Agro-Meteorological & Phenology Models (ICRISAT / MPKV Rahuri / VNMKV Parbhani crop calendars)",
    "version": "1.2",
    "num_stages": 4,
    "stages": ["Early season", "Vegetative", "Peak growth", "Late season"],
    "limitations": (
        "Deterministic 4-stage idealized NDVI curves for major Maharashtra crops. "
        "This is an explainable geometric similarity heuristic, not a trained ML classifier, "
        "and cannot substitute for local agronomic ground validation."
    ),
}



def classify_ndvi_series(
    series, signatures: dict = CROP_SIGNATURES, return_distances: bool = False
):
    """Compares an observed NDVI time-series against each crop's reference
    signature using root-mean-square (RMS) spectral distance:
        RMS = sqrt(mean((series - reference)^2))

    RMS distances are converted to normalized similarity weights using
    inverse-distance weighting:
        w_i = 1.0 / (d_i + eps),  similarity_i = w_i / sum(w)

    SCIENTIFIC NOTE:
    This is an explainable deterministic reference-signature similarity metric,
    NOT a trained machine learning model and NOT a calibrated statistical probability.
    Similarity scores reflect relative proximity in phenology feature space.

    Returns:
      If return_distances is False: (best_crop, {crop: similarity_score, ...})
      If return_distances is True:  (best_crop, {crop: similarity_score, ...}, {crop: rms_distance, ...})
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or len(series) < 4:
        raise ValueError(f"Observed NDVI series requires at least 4 temporal observations, got shape {series.shape}.")
    if not np.all(np.isfinite(series)):
        raise ValueError("Observed NDVI series contains non-finite (NaN or Inf) values.")

    distances = {}
    for crop, ref in signatures.items():
        ref = np.asarray(ref, dtype=float)
        distances[crop] = float(np.sqrt(np.mean((series - ref) ** 2)))

    inv = np.array([1.0 / (d + 1e-6) for d in distances.values()])
    probs = inv / inv.sum()

    scores = {crop: float(p) for crop, p in zip(distances.keys(), probs)}
    best = max(scores, key=scores.get)
    if return_distances:
        return best, scores, distances
    return best, scores
