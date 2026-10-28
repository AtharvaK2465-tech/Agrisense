"""
Vision Agent
------------
Classifies the clipped imagery into a stress category. Ships with a
transparent heuristic over the NDVI/NDMI indices so the pipeline is fully
runnable and explainable out of the box; swap in a trained PyTorch model
(see `_model_predict`) once you have labeled training data.
"""
import numpy as np

from agents.state import PipelineState

LABELS = ["healthy", "water_stress", "nutrient_deficiency", "disease", "no_crop"]


def _heuristic_classify(
    ndvi: np.ndarray,
    ndmi: np.ndarray = None,
    is_non_agri: bool = False,
    crop_ndvi: np.ndarray = None,
    water_fraction_pct: float = 0.0,
) -> tuple[str, float]:
    """A transparent, explainable stand-in classifier based on mean index values
    and how much of the scene falls below a stress threshold.
    
    Scientific Safeguards:
      - Water pixels (negative NDVI) are strictly excluded from crop stress fractions.
      - A low NDVI observation alone (e.g. from early emergence or bare fallow soil)
        does NOT establish disease, drought or crop failure.
      - Disease is only suggested when localized, patchy canopy depression is observed.
    """
    if is_non_agri:
        return "no_crop", 0.95

    target_ndvi = crop_ndvi if crop_ndvi is not None else ndvi
    if target_ndvi is None:
        return "unavailable", 0.0

    usable_finite = target_ndvi[np.isfinite(target_ndvi)]
    # Ensure water (negative NDVI) never enters agricultural canopy stress statistics
    usable_crop_ndvi = usable_finite[usable_finite >= 0.0]

    if len(usable_crop_ndvi) == 0:
        if water_fraction_pct > 50.0:
            return "no_crop", 0.95
        return "unavailable", 0.0

    mean_ndvi = float(np.mean(usable_crop_ndvi))
    # Stressed fraction: agricultural canopy pixels displaying low NDVI (< 0.3)
    stressed_fraction = float(np.mean(usable_crop_ndvi < 0.3))

    if ndmi is not None:
        usable_ndmi = ndmi[np.isfinite(ndmi)]
        # When crop_ndvi is provided, align NDMI with crop-usable pixels
        if crop_ndvi is not None:
            crop_mask = np.isfinite(crop_ndvi) & (crop_ndvi >= 0.0)
            if crop_mask.shape == ndmi.shape:
                usable_crop_ndmi = ndmi[crop_mask]
            else:
                usable_crop_ndmi = usable_ndmi
        else:
            usable_crop_ndmi = usable_ndmi
        mean_ndmi = float(np.mean(usable_crop_ndmi)) if len(usable_crop_ndmi) > 0 else None
    else:
        mean_ndmi = None

    std_ndvi = float(np.std(usable_crop_ndvi)) if len(usable_crop_ndvi) > 1 else 0.0

    if stressed_fraction < 0.05:
        return "healthy", round(float(0.9 - stressed_fraction), 3)
    if mean_ndmi is not None and mean_ndmi < 0.1:
        return "water_stress", round(float(min(0.6 + stressed_fraction, 0.95)), 3)
    # Scientific safeguard: low NDVI alone does not establish disease.
    # Disease requires localized patchy depression (canopy standard deviation >= 0.04).
    # Uniform low NDVI represents low vigor, bare fallow soil, or early emergence.
    if mean_ndvi < 0.2 and stressed_fraction >= 0.20 and std_ndvi >= 0.04:
        return "disease", round(float(min(0.55 + stressed_fraction, 0.9)), 3)
    return "nutrient_deficiency", round(float(min(0.5 + stressed_fraction, 0.85)), 3)


def _model_predict(raster: np.ndarray) -> tuple[str, float]:
    """
    LIVE MODE placeholder — plug in your trained PyTorch/TensorFlow model.

        model = load_model("models/crop_stress_cnn.pt")
        tensor = preprocess(raster)  # normalize, resize, to CHW tensor
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)
        label = LABELS[probs.argmax()]
        confidence = float(probs.max())
        return label, confidence
    """
    raise NotImplementedError(
        "TODO: load a trained model checkpoint and run inference here."
    )


def vision_agent(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    if state.get("crop_decision_status") == "unavailable":
        return {
            **state,
            "vision_classification": "unavailable",
            "vision_confidence": 0.0,
        }

    is_non_agri = bool(state.get("is_non_agri", False)) or (state.get("crop_type") == "no_crop") or (state.get("crop_decision_status") == "no_crop")
    water_pct = state.get("quality_metrics", {}).get("water_fraction_pct", 0.0)
    label, confidence = _heuristic_classify(
        state.get("ndvi"),
        state.get("ndmi"),
        is_non_agri=is_non_agri,
        crop_ndvi=state.get("crop_ndvi"),
        water_fraction_pct=water_pct,
    )

    return {
        **state,
        "vision_classification": label,
        "vision_confidence": round(confidence, 3),
    }

