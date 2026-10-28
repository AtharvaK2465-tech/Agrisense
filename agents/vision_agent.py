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


def _heuristic_classify(ndvi: np.ndarray, ndmi: np.ndarray = None, is_non_agri: bool = False) -> tuple[str, float]:
    """A simple, explainable stand-in classifier based on mean index values
    and how much of the scene falls below a stress threshold. Replace with
    `_model_predict` once a trained CNN is available."""
    if is_non_agri:
        return "no_crop", 0.95

    if ndvi is None:
        return "unavailable", 0.0

    usable_ndvi = ndvi[np.isfinite(ndvi)]
    if len(usable_ndvi) == 0:
        return "unavailable", 0.0

    mean_ndvi = float(np.mean(usable_ndvi))
    stressed_fraction = float(np.mean(usable_ndvi < 0.3))

    if ndmi is not None:
        usable_ndmi = ndmi[np.isfinite(ndmi)]
        mean_ndmi = float(np.mean(usable_ndmi)) if len(usable_ndmi) > 0 else None
    else:
        mean_ndmi = None

    if stressed_fraction < 0.05:
        return "healthy", round(float(0.9 - stressed_fraction), 3)
    if mean_ndmi is not None and mean_ndmi < 0.1:
        return "water_stress", round(float(min(0.6 + stressed_fraction, 0.95)), 3)
    if mean_ndvi < 0.2:
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
    label, confidence = _heuristic_classify(state.get("ndvi"), state.get("ndmi"), is_non_agri=is_non_agri)

    return {
        **state,
        "vision_classification": label,
        "vision_confidence": round(confidence, 3),
    }

