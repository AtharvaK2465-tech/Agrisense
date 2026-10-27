"""
Reasoning Agent
---------------
Combines the Vision Agent's classification with regional rainfall data to
evaluate competing agronomic explanations (e.g. distinguishing potential
drought-related canopy stress from non-drought stress) under strict scientific
safeguards: weather is contextual evidence, not proof of disease causality.
"""
from shapely.geometry import shape

from agents.state import PipelineState
from utils.weather import get_recent_weather


def _polygon_centroid(geojson: dict) -> tuple[float, float]:
    geom = shape(geojson)
    c = geom.centroid
    return c.y, c.x  # lat, lon


def reasoning_agent(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    lat, lon = _polygon_centroid(state["polygon_geojson"])
    weather = get_recent_weather(lat, lon, state["date_end"])

    classification = state["vision_classification"]
    confidence = state["vision_confidence"]
    rainfall = weather.get("total_rainfall_mm")

    reasoning_lines = [
        f"Vision Agent observed '{classification}' pattern with {confidence:.0%} consistency score."
    ]

    diagnosis = classification
    risk_level = "low"

    if classification == "no_crop" or state.get("crop_type") == "no_crop" or state.get("is_non_agri"):
        diagnosis = "No Crop Detected (Non-Agricultural Ground)"
        risk_level = "low"
        ground_ctx = state.get("ground_context") or "Non-agricultural / built-up / campus ground"
        reasoning_lines.append(
            f"The analyzed location was identified as non-agricultural terrain ({ground_ctx}). "
            "Flat spectral reflectance corresponds to artificial turf, educational campus ground, pavement, or built-up soil, "
            "rather than crop failure or disease. No agricultural intervention is required."
        )
    elif classification == "water_stress" and rainfall is not None:
        if rainfall < 10:
            reasoning_lines.append(
                f"Low precipitation ({rainfall}mm recorded in the past {weather['days']} days) provides environmental "
                "context consistent with potential canopy moisture deficit. Verify root-zone soil moisture in field "
                "before altering irrigation schedules."
            )
            diagnosis = "Moisture Deficit Stress Pattern"
            risk_level = "high"
        else:
            reasoning_lines.append(
                f"Recent precipitation ({rainfall}mm in {weather['days']} days) indicates adequate ambient moisture. "
                "Observed canopy depression is therefore less likely to be a simple moisture deficit, and may reflect "
                "drainage issues, root waterlogging, or pest/foliar pressure. Physical field inspection recommended."
            )
            diagnosis = "Canopy Stress (Non-Drought Pattern)"
            risk_level = "high"
    elif classification == "disease":
        reasoning_lines.append(
            "Spectral index pattern indicates localized canopy depression consistent with vegetative stress. "
            "Satellite observations cannot identify specific biological pathogens or pests; in-field scouting "
            "and laboratory diagnostic sampling are required for confirmation."
        )
        diagnosis = "Vegetative Stress Pattern (Unconfirmed Pathogen)"
        risk_level = "high"
    elif classification == "nutrient_deficiency":
        reasoning_lines.append(
            "Moderate NDVI and red-edge depression observed, consistent with variation in canopy chlorophyll vigor. "
            "Soil testing and tissue nutrient analysis are recommended before applying fertilizers."
        )
        diagnosis = "Chlorophyll / Canopy Vigor Depression"
        risk_level = "medium"
    else:
        reasoning_lines.append(
            "Spectral vegetation indices fall within normal vigorous ranges for cultivated parcels in this region."
        )
        diagnosis = "Healthy Canopy"
        risk_level = "low"

    if state.get("crop_decision_status") == "uncertain" or state.get("crop_type") == "uncertain":
        reasoning_lines.append(
            f"Crop Identification Status: {state.get('crop_decision_reason', 'Agricultural land detected, but crop species is uncertain.')} "
            "Vegetation condition is reported as general canopy vigor; crop-specific disease thresholds require ground verification."
        )

    return {
        **state,
        "weather": weather,
        "diagnosis": diagnosis,
        "diagnosis_reasoning": " ".join(reasoning_lines),
        "risk_level": risk_level,
    }
