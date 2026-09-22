"""
Report Agent
------------
Turns the Reasoning Agent's structured diagnosis (plus the Crop Agent's
crop identification and district/baseline context) into a readable
report. Uses Gemini when GEMINI_API_KEY is set; otherwise falls back to a
clean template so the app still produces a full report in mock mode.
"""
from agents.state import PipelineState
from utils.config import GEMINI_MOCK, GEMINI_API_KEY
from utils.crops import CROP_DISPLAY_NAMES
from utils.advisories import match_advisories


def _advisory_matches(state: PipelineState) -> list[dict]:
    return match_advisories(
        state.get("district", ""), state.get("crop_type", ""), state.get("season", "")
    )


def _advisory_text(matches: list[dict]) -> str:
    if not matches:
        return "No matching demo advisory records for this district/crop/season."
    return "\n".join(f"- {m['issue']} ({m['risk']} risk) — {m['note']}" for m in matches)


def _mock_report(state: PipelineState, advisory_matches: list[dict]) -> str:
    w = state.get("weather", {})
    crop_name = CROP_DISPLAY_NAMES.get(state.get("crop_type", ""), state.get("crop_type", "Unknown"))
    current_ndvi = state.get("baseline_ndvi", 0) + state.get("ndvi_anomaly", 0)

    return f"""## Crop Health Report — Selected Area

**Diagnosis:** {state['diagnosis'].replace('_', ' ').title()}
**Risk level:** {state['risk_level'].upper()}
**Vision confidence:** {state['vision_confidence']:.0%}
**Crop type:** {crop_name} ({state.get('crop_confidence', 0):.0%} confidence)
**District / season:** {state.get('district', 'Unknown')} / {state.get('season', 'Unknown').title()}

### Reasoning
{state['diagnosis_reasoning']}

### Crop identification
{state.get('crop_agent_notes', 'N/A')}

### Historical baseline
Current mean NDVI is {current_ndvi:.2f} vs. a 3-year seasonal baseline of
{state.get('baseline_ndvi', 0):.2f} for this area (anomaly: {state.get('ndvi_anomaly', 0):+.2f}).

### Supporting data
- Mean NDVI-derived condition consistent with the vision classification above
- Recent rainfall: {w.get('total_rainfall_mm', 'N/A')} mm over {w.get('days', '?')} days
- Cloud cover at capture: {state.get('cloud_cover_pct', 0):.1f}%

### Known issues in this district/crop/season (demo reference data)
{_advisory_text(advisory_matches)}

### Recommendation
{"Immediate field inspection recommended." if state['risk_level'] == 'high'
 else "Continue routine monitoring; re-check in 1-2 weeks." if state['risk_level'] == 'medium'
 else "No action needed — area appears healthy."}

*Note: this report was generated in MOCK MODE (no GEMINI_API_KEY set). Add a real key
to .env to generate natural-language reports via the Gemini API.*
"""


def _real_report(state: PipelineState, advisory_matches: list[dict]) -> str:
    """LIVE MODE — calls Gemini to turn the structured diagnosis into a
    short, plain-language report. Raises on failure so report_agent()
    below can fall back to the template rather than crash the pipeline."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-3.6-flash")

    w = state.get("weather", {})
    crop_name = CROP_DISPLAY_NAMES.get(state.get("crop_type", ""), state.get("crop_type", "Unknown"))
    prompt = f"""Write a short, clear crop health report for a farmer/agronomist
based on this structured diagnosis for a selected area in Maharashtra, India.

Diagnosis: {state['diagnosis']}
Risk level: {state['risk_level']}
Vision model confidence: {state['vision_confidence']:.0%}
Reasoning: {state['diagnosis_reasoning']}
Crop type: {crop_name} ({state.get('crop_confidence', 0):.0%} confidence)
District / season: {state.get('district', 'Unknown')} / {state.get('season', 'Unknown')}
Current NDVI vs. 3-year seasonal baseline: anomaly {state.get('ndvi_anomaly', 0):+.2f}
Recent rainfall: {w.get('total_rainfall_mm', 'unknown')} mm over {w.get('days', '?')} days
Cloud cover at image capture: {state.get('cloud_cover_pct', 0):.1f}%
Known local advisories: {_advisory_text(advisory_matches)}

Structure it with a one-line diagnosis summary, a short "why" explanation
in plain language (avoid jargon like NDVI), and one concrete recommended
action. Keep the whole thing under 200 words. Format as Markdown with
headers matching: ## Crop Health Report, ### Diagnosis, ### Why,
### Recommendation."""

    response = model.generate_content(prompt)
    return response.text


def report_agent(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return {**state, "report_markdown": f"Could not generate report: {state['error']}"}

    advisory_matches = _advisory_matches(state)

    if GEMINI_MOCK:
        report = _mock_report(state, advisory_matches)
    else:
        try:
            report = _real_report(state, advisory_matches)
        except Exception as e:
            # Live call failed (quota, network, bad key, etc.) — don't let
            # a report-generation hiccup take down the whole pipeline.
            report = _mock_report(state, advisory_matches) + f"\n\n*(Live Gemini call failed, showed template instead: {e})*"

    if state.get("geo_fallback_note"):
        report += f"\n\n*(Note: {state['geo_fallback_note']})*"

    return {**state, "report_markdown": report, "advisory_matches": advisory_matches}
