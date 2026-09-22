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
}

CROP_DISPLAY_NAMES = {
    "sugarcane": "Sugarcane",
    "cotton": "Cotton",
    "soybean": "Soybean",
    "jowar": "Jowar (Sorghum)",
    "tur": "Tur (Pigeon Pea)",
    "fallow": "Fallow / Bare Soil",
}

# Hex colors for the map legend / overlay, loosely following standard
# agronomy map conventions (deep green = perennial cane, warm tones for
# the shorter kharif crops, brown = bare/fallow).
CROP_COLORS = {
    "sugarcane": "#1b7837",
    "cotton": "#fee08b",
    "soybean": "#f1a340",
    "jowar": "#d73027",
    "tur": "#66c2a5",
    "fallow": "#8c6d31",
}


def classify_ndvi_series(series, signatures: dict = CROP_SIGNATURES) -> tuple[str, dict]:
    """Compares an observed NDVI time-series against each crop's reference
    signature using root-mean-square distance, then converts distances to
    a normalized confidence score via inverse-distance weighting — a
    simple, fully explainable stand-in for a trained 1-NN/DTW classifier.

    Returns (best_crop, {crop: confidence_score, ...}) where scores sum to 1.
    """
    series = np.asarray(series, dtype=float)
    distances = {}
    for crop, ref in signatures.items():
        ref = np.asarray(ref, dtype=float)
        distances[crop] = float(np.sqrt(np.mean((series - ref) ** 2)))

    inv = np.array([1.0 / (d + 1e-6) for d in distances.values()])
    probs = inv / inv.sum()

    scores = {crop: float(p) for crop, p in zip(distances.keys(), probs)}
    best = max(scores, key=scores.get)
    return best, scores
