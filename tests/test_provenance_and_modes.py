"""
Testing explicit Live Mode and Mock Mode separation and provenance tracking.
Ensures that:
1. Mock mode is explicitly labeled with synthetic raster metadata.
2. Missing credentials in live mode without fallback raises an explicit error.
3. Fallback mode is explicitly recorded as live_fallback_to_mock with unmissable warnings.
4. Gemini live vs template fallback is tracked in provenance.
5. Never silently substitutes synthetic data while presenting as live satellite data.
"""
import pytest
from datetime import date
import utils.config as config
from agents.geo_agent import geo_agent
from agents.report_agent import report_agent
from agents.state import PipelineState

TEST_POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [77.10, 19.70], [77.15, 19.70], [77.15, 19.75], [77.10, 19.75], [77.10, 19.70]
    ]],
}


def test_mock_mode_explicit_provenance(monkeypatch):
    monkeypatch.setattr(config, "SENTINEL_HUB_MOCK", True)
    state: PipelineState = {
        "polygon_geojson": TEST_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
    }
    out = geo_agent(state)
    assert "error" not in out
    prov = out["provenance"]
    assert prov["execution_mode"] == "mock"
    assert prov["is_synthetic_imagery"] is True
    assert "Synthetic" in prov["provider"]
    assert "SYNTH" in prov["scene_id"]


def test_live_mode_failure_without_fallback_yields_hard_error(monkeypatch):
    # Set to Live mode (MOCK=False), but with invalid credentials and fallback disabled
    monkeypatch.setattr(config, "SENTINEL_HUB_MOCK", False)
    monkeypatch.setattr(config, "SENTINEL_HUB_CLIENT_ID", "")
    monkeypatch.setattr(config, "SENTINEL_HUB_CLIENT_SECRET", "")
    monkeypatch.setattr(config, "ALLOW_LIVE_FALLBACK_TO_MOCK", False)

    state: PipelineState = {
        "polygon_geojson": TEST_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
    }
    out = geo_agent(state)
    assert "error" in out
    assert "Live Sentinel Hub imagery retrieval failed" in out["error"]


def test_live_mode_failure_with_fallback_records_explicit_fallback_provenance(monkeypatch):
    monkeypatch.setattr(config, "SENTINEL_HUB_MOCK", False)
    monkeypatch.setattr(config, "SENTINEL_HUB_CLIENT_ID", "")
    monkeypatch.setattr(config, "SENTINEL_HUB_CLIENT_SECRET", "")
    monkeypatch.setattr(config, "ALLOW_LIVE_FALLBACK_TO_MOCK", True)

    state: PipelineState = {
        "polygon_geojson": TEST_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
    }
    out = geo_agent(state)
    assert "error" not in out
    prov = out["provenance"]
    assert prov["execution_mode"] == "live_fallback_to_mock"
    assert prov["is_synthetic_imagery"] is True
    assert "geo_fallback_note" in out
    assert "NOTICE: Live Sentinel Hub retrieval failed" in out["geo_fallback_note"]


def test_report_provenance_distinguishes_gemini_vs_template(monkeypatch):
    # Force Mock Gemini
    monkeypatch.setattr(config, "is_gemini_mock", lambda: True)

    dummy_state: PipelineState = {
        "polygon_geojson": TEST_POLYGON,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
        "diagnosis": "Healthy Canopy",
        "diagnosis_reasoning": "Vigorous NDVI observed.",
        "risk_level": "low",
        "vision_confidence": 0.9,
        "provenance": {"execution_mode": "mock", "is_synthetic_imagery": True},
    }
    out = report_agent(dummy_state)
    assert out["provenance"]["report_source"] == "mock-template"
    assert "MOCK MODE" in out["report_markdown"]
