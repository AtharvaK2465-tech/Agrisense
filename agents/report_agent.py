"""
Report Agent
------------
Turns the Reasoning Agent's structured diagnosis, the Crop Agent's crop/baseline
evidence, and Geo Agent's provenance metadata into a transparent, scientifically
hardened report. Uses Gemini when GEMINI_API_KEY is configured, or falls back to
an agronomic deterministic template. Reports are strictly validated by
agents/report_validator.py to prevent ungrounded claims or chemical directives.
"""
from typing import Dict, Any, List
from agents.state import PipelineState
from agents.report_validator import validate_report, sanitize_report_violations
import utils.config as config
from utils.crops import CROP_DISPLAY_NAMES
from utils.advisories import match_advisories


def _advisory_matches(state: PipelineState) -> List[Dict[str, Any]]:
    return match_advisories(
        state.get("district", ""), state.get("crop_type", ""), state.get("season", "")
    )


def _advisory_text(matches: List[Dict[str, Any]]) -> str:
    if not matches:
        return "No matching regional advisory records for this district/crop/season."
    return "\n".join(f"- {m['issue']} ({m['risk']} risk) — {m['note']}" for m in matches)


def _format_indices_summary(state: PipelineState) -> str:
    quality = state.get("indices_quality", {})
    lines = []
    # NDVI
    ndvi_q = quality.get("ndvi", {})
    lines.append(f"- **NDVI:** {ndvi_q.get('status', 'calculated').title()} ({ndvi_q.get('formula', '(NIR-Red)/(NIR+Red)')})")

    # NDMI
    ndmi_q = quality.get("ndmi", {})
    if ndmi_q.get("status") == "calculated":
        lines.append(f"- **NDMI:** Calculated ({ndmi_q.get('bands', 'B08, B11')})")
    else:
        lines.append(f"- **NDMI:** Unavailable ({ndmi_q.get('reason', 'SWIR band not available')})")

    # NDRE
    ndre_q = quality.get("ndre", {})
    if ndre_q.get("status") == "calculated":
        lines.append(f"- **NDRE:** Calculated ({ndre_q.get('bands', 'B08, B05')})")
    else:
        lines.append(f"- **NDRE:** Unavailable ({ndre_q.get('reason', 'Red-edge band not available')})")

    return "\n".join(lines)


def _mock_report(state: PipelineState, advisory_matches: List[Dict[str, Any]], is_live_failure: bool = False) -> str:
    w = state.get("weather", {})
    prov = state.get("provenance", {})
    crop_name = CROP_DISPLAY_NAMES.get(state.get("crop_type", ""), state.get("crop_type", "Unknown"))

    surface = state.get("surface_fractions", {})
    veg_pct = surface.get("vegetation", 0.0) * 100
    bare_pct = surface.get("bare_surface", 0.0) * 100
    water_pct = surface.get("water_surface", 0.0) * 100

    # Baseline text
    baseline_status = state.get("baseline_status", "unavailable")
    if baseline_status == "synthetic_simulation":
        baseline_text = (
            f"Current mean NDVI is {state.get('baseline_ndvi', 0) + state.get('ndvi_anomaly', 0):.2f} "
            f"vs. simulated 3-year baseline {state.get('baseline_ndvi', 0):.2f} (anomaly: {state.get('ndvi_anomaly', 0):+.2f}). "
            f"*(Note: Simulated baseline for demonstration; multi-year satellite archive query not executed).* "
        )
    elif baseline_status == "historical_actual":
        baseline_text = (
            f"Current mean NDVI vs. actual 3-year seasonal baseline: {state.get('baseline_ndvi', 0):.2f} "
            f"(anomaly: {state.get('ndvi_anomaly', 0):+.2f})."
        )
    else:
        baseline_text = f"Baseline unavailable: {state.get('baseline_reason', 'Historical multi-year observations not retrieved')}."

    if is_live_failure:
        footer = "*Note: Generated via deterministic fallback template because live LLM call was unavailable.*"
    elif prov.get("is_synthetic_imagery"):
        footer = "*Note: Generated in DEMO / MOCK MODE using synthetic raster data and deterministic agronomic template.*"
    else:
        footer = "*Note: Generated using deterministic template on live Copernicus Sentinel-2 observations.*"

    if state.get("crop_type") == "no_crop" or state.get("is_non_agri") or "No Crop" in state.get("diagnosis", ""):
        rec_text = (
            "No agricultural intervention required. The selected boundary corresponds to non-agricultural "
            "or built-up surface (educational campus ground, residential, industrial, or open water) where crops "
            "are not under commercial cultivation."
        )
        header_title = "Land Assessment Report — Non-Agricultural Ground"
    elif state.get("crop_type") == "uncertain":
        rec_text = (
            "Conduct physical ground survey across the parcel to verify crop identity and emergence stage. "
            "Active vegetative canopy is observed, but multi-temporal satellite phenology alone cannot distinguish "
            "between regional crop candidates with sufficient discrimination margin. Do not apply crop-specific treatments."
        )
        header_title = "Crop Health Report — Agricultural Land (Species Uncertain)"
    elif state["risk_level"] == "high":
        rec_text = (
            "Field scouting recommended: inspect root-zone soil moisture and foliage across stressed patches. "
            "Consult a local agricultural extension officer (Krishi Vigyan Kendra) before applying any irrigation changes or crop treatments."
        )
        header_title = "Crop Health & Stress Assessment Report"
    elif state["risk_level"] == "medium":
        rec_text = (
            "Routine field monitoring advised: observe canopy color and growth over the next 7-10 days. "
            "Perform soil and leaf tissue testing if nutrient deficiency symptoms persist."
        )
        header_title = "Crop Health & Stress Assessment Report"
    else:
        rec_text = (
            "Canopy indices indicate vigorous vegetative condition. Maintain standard cultivation practices and routine field scouting."
        )
        header_title = "Crop Health & Stress Assessment Report"

    return f"""## {header_title}

**Observed Diagnosis:** {state['diagnosis'].replace('_', ' ').title()}
**Assessed Risk Level:** {state['risk_level'].upper()}
**Vision Consistency Score:** {state['vision_confidence']:.0%}
**Identified Surface / Crop:** {crop_name} ({state.get('crop_confidence', 0):.0%} match score)
**District / Season:** {state.get('district', 'Unknown')} / {state.get('season', 'Unknown').title()}

### Scientific Reasoning & Environmental Context
{state['diagnosis_reasoning']}

### Crop & Land Identification Evidence
{state.get('crop_agent_notes', 'N/A')}

### Spectral Indices & Surface Breakdown
{_format_indices_summary(state)}
- **Surface Fractions:** Active Vegetation: {veg_pct:.1f}% | Bare Ground/Fallow: {bare_pct:.1f}% | Water: {water_pct:.1f}%
- **Valid Pixel Coverage:** {state.get('valid_pixels_pct', 100.0):.1f}%

### Temporal Baseline Context
{baseline_text}

### Observation Provenance
- **Data Source:** {prov.get('provider', 'Sentinel-2 L2A / Synthetic')} ({'Synthetic Demonstration' if prov.get('is_synthetic_imagery') else 'Live Satellite Observation'})
- **Scene Identifier:** {prov.get('scene_id', 'N/A')}
- **Acquisition Timestamp:** {prov.get('acquisition_date', 'N/A')}
- **Cloud Cover Estimate:** {state.get('cloud_cover_pct', 0):.1f}% (Acceptance threshold: {prov.get('cloud_threshold_pct', 30)}%)
- **Recent Rainfall Context:** {w.get('total_rainfall_mm', 'N/A')} mm over {w.get('days', '?')} days ({w.get('source', 'Open-Meteo')})

### Regional Demonstration Advisories
{_advisory_text(advisory_matches)}

### Recommended Action
{rec_text}

{footer}
"""


def _real_report(state: PipelineState, advisory_matches: List[Dict[str, Any]]) -> str:
    """LIVE MODE — queries Gemini with strict scientific guidelines: no chemical directives,
    no definitive disease assertions without physical testing, and mandatory ground verification."""
    import google.generativeai as genai
    import os

    api_key = config.get_gemini_api_key()
    genai.configure(api_key=api_key, transport="rest")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)

    w = state.get("weather", {})
    crop_name = CROP_DISPLAY_NAMES.get(state.get("crop_type", ""), state.get("crop_type", "Unknown"))
    current_ndvi = (state.get("baseline_ndvi") or 0.0) + (state.get("ndvi_anomaly") or 0.0)

    prompt = f"""You are a scientific remote sensing and agronomic report writer.
Write a concise, rigorous crop assessment report for a parcel in Maharashtra, India based on satellite observations.

CRITICAL SCIENTIFIC SAFETY RULES:
1. NEVER prescribe chemical pesticides, fungicides, herbicides, or specific fertilizer dosing.
2. Distinguish satellite spectral observations from in-field biological reality: satellite indices detect canopy vigor anomalies, NOT confirmed pathogens or pests.
3. Rain and weather are environmental context, NOT proof of pathogen infection or crop loss.
4. Recommendations must focus on physical field inspection, soil testing, and consulting local Krishi Vigyan Kendra (KVK) extension officers.
5. Keep under 220 words. Format cleanly in Markdown with headers: ## Crop Health Report, ### Diagnosis, ### Scientific Reasoning, ### Field Verification Recommendation.

Data:
- Surface / Crop: {crop_name} ({state.get('crop_confidence', 0):.0%} score)
- Diagnosis: {state['diagnosis']}
- Risk Level: {state['risk_level']}
- Reasoning: {state['diagnosis_reasoning']}
- District / Season: {state.get('district', 'Unknown')} / {state.get('season', 'Unknown')}
- Rainfall context: {w.get('total_rainfall_mm', 'N/A')} mm in {w.get('days', 14)} days
- Cloud Cover: {state.get('cloud_cover_pct', 0):.1f}%
- Advisories Context: {_advisory_text(advisory_matches)}"""

    response = model.generate_content(prompt)
    return response.text


def report_agent(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return {**state, "report_markdown": f"Could not generate report: {state['error']}"}

    advisory_matches = _advisory_matches(state)
    report_source = "deterministic-template"

    if config.is_gemini_mock():
        raw_report = _mock_report(state, advisory_matches, is_live_failure=False)
        report_source = "mock-template"
    else:
        try:
            raw_report = _real_report(state, advisory_matches)
            report_source = "gemini-live"
        except Exception as e:
            raw_report = _mock_report(state, advisory_matches, is_live_failure=True)
            report_source = "gemini-fallback-template"
            raw_report += f"\n\n*(Note: Live Gemini report generation failed ({e}); displayed scientific template).* "

    if state.get("geo_fallback_note"):
        raw_report += f"\n\n*(Note: {state['geo_fallback_note']})*"

    # Validate against scientific language rules
    validation = validate_report(raw_report, state)
    if not validation["is_valid"]:
        # If Gemini generation violates scientific safeguards, fall back to compliant deterministic template
        template_report = _mock_report(state, advisory_matches, is_live_failure=True)
        template_report += (
            "\n\n*(Note: Live LLM-generated text was rejected by the Scientific Report Validator "
            f"({'; '.join(validation['violations'])}) and safely substituted with verified agronomic template).*"
        )
        raw_report = template_report
        report_source = f"{report_source}-safeguard-fallback"
        validation = validate_report(raw_report, state)

    provenance = dict(state.get("provenance", {}))
    provenance["report_source"] = report_source
    provenance["report_validation_status"] = validation["status"]

    return {
        **state,
        "report_markdown": raw_report,
        "report_validation": validation,
        "advisory_matches": advisory_matches,
        "provenance": provenance,
    }
