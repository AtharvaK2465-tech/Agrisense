"""
The shared state object that flows through every node in the LangGraph
pipeline. Each agent reads what it needs and writes its output back in.
"""
from typing import TypedDict, Optional, Any
from datetime import date


class PipelineState(TypedDict, total=False):
    # --- input from the UI ---
    polygon_geojson: dict
    date_start: date
    date_end: date

    # --- Geo Agent output ---
    raster: Optional[Any]          # clipped multi-band array (mock or real)
    ndvi: Optional[Any]
    ndmi: Optional[Any]
    ndre: Optional[Any]
    indices_quality: Optional[dict]  # status and availability metadata for ndvi, ndmi, ndre
    surface_fractions: Optional[dict]  # vegetation, bare_soil, water, unclassified fractions
    valid_pixels_pct: Optional[float]
    cloud_cover_pct: Optional[float]
    query_attempts: int            # tracks re-query loop count
    geo_fallback_note: Optional[str]  # set if a live Sentinel Hub call failed and fell back to mock

    # --- Provenance & Data Source Tracking ---
    provenance: Optional[dict]     # comprehensive source, mode, scene metadata, and verification trail

    # --- Input/Validation Agent output ---
    imagery_ok: Optional[bool]
    validation_notes: Optional[str]

    # --- Crop-Type Agent output ---
    crop_type: Optional[str]
    crop_confidence: Optional[float]
    crop_scores: Optional[dict]            # {crop: confidence} for every candidate
    crop_confidence_map: Optional[Any]     # per-pixel confidence array, same shape as ndvi
    phenology_stage_labels: Optional[list]
    ndvi_timeseries: Optional[list]        # NDVI mean at each phenology stage
    district: Optional[str]
    season: Optional[str]                  # kharif / rabi
    baseline_ndvi: Optional[float]         # 3-year seasonal average for this area
    baseline_status: Optional[str]         # "historical_actual" | "synthetic_simulation" | "unavailable"
    baseline_reason: Optional[str]
    ndvi_anomaly: Optional[float]          # current mean NDVI - baseline
    crop_agent_notes: Optional[str]
    is_non_agri: Optional[bool]
    ground_context: Optional[str]
    agricultural_area: Optional[bool]
    agricultural_confidence: Optional[float]
    crop_decision_status: Optional[str]    # "no_crop" | "identified" | "uncertain"
    crop_decision_reason: Optional[str]
    crop_similarity_margin: Optional[float]
    crop_evidence: Optional[dict]

    # --- Vision Agent output ---
    vision_classification: Optional[str]   # disease / water_stress / nutrient_deficiency / healthy / no_crop
    vision_confidence: Optional[float]

    # --- weather fusion (Reasoning Agent input) ---
    weather: Optional[dict]

    # --- Reasoning Agent output ---
    diagnosis: Optional[str]
    diagnosis_reasoning: Optional[str]
    risk_level: Optional[str]      # low / medium / high

    # --- Report Agent output ---
    report_markdown: Optional[str]
    report_validation: Optional[dict]      # scientific validation checks outcome
    advisory_matches: Optional[list]       # demo ground-truth advisories matching district/crop/season

    # --- control flow ---
    error: Optional[str]
