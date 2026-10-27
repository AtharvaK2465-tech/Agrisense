"""
Unit tests for weather retrieval and parsing robustness.
Tests network failure, malformed responses, missing observations, and empty rainfall windows.
"""
from datetime import date
from unittest.mock import patch, MagicMock
import numpy as np

from utils.weather import get_recent_weather
from agents.reasoning_agent import reasoning_agent


def test_get_recent_weather_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "daily": {
            "precipitation_sum": [0.0, 2.5, 0.0, 1.2, 0.0, 4.0, 0.0],
            "temperature_2m_max": [32.0, 31.0, 30.5, 29.0, 30.0, 28.5, 31.0],
        }
    }
    with patch("requests.get", return_value=mock_resp):
        out = get_recent_weather(19.5, 76.5, date(2026, 9, 15), days=7)
        assert out["total_rainfall_mm"] == 7.7
        assert out["days"] == 7
        assert out["observations_count"] == 7
        assert out["source"] == "open-meteo"


def test_get_recent_weather_handles_network_timeout():
    with patch("requests.get", side_effect=Exception("Connection timed out")):
        out = get_recent_weather(19.5, 76.5, date(2026, 9, 15), days=7)
        assert out["total_rainfall_mm"] is None
        assert "unavailable" in out["source"]
        assert "Connection timed out" in out["source"]


def test_get_recent_weather_handles_malformed_non_dict_response():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = ["not", "a", "dict"]
    with patch("requests.get", return_value=mock_resp):
        out = get_recent_weather(19.5, 76.5, date(2026, 9, 15), days=7)
        assert out["total_rainfall_mm"] is None
        assert "malformed response format" in out["source"]


def test_get_recent_weather_handles_missing_daily_block():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"hourly": {}}
    with patch("requests.get", return_value=mock_resp):
        out = get_recent_weather(19.5, 76.5, date(2026, 9, 15), days=7)
        assert out["total_rainfall_mm"] is None
        assert "missing daily block" in out["source"]


def test_get_recent_weather_handles_empty_rainfall_window():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"daily": {"precipitation_sum": []}}
    with patch("requests.get", return_value=mock_resp):
        out = get_recent_weather(19.5, 76.5, date(2026, 9, 15), days=7)
        assert out["total_rainfall_mm"] is None
        assert "empty precipitation observation window" in out["source"]


def test_get_recent_weather_handles_non_numeric_values():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "daily": {
            "precipitation_sum": [None, "null", 3.5, None, 1.5]
        }
    }
    with patch("requests.get", return_value=mock_resp):
        out = get_recent_weather(19.5, 76.5, date(2026, 9, 15), days=5)
        assert out["total_rainfall_mm"] == 5.0
        assert out["observations_count"] == 2


def test_reasoning_agent_water_stress_with_unavailable_weather():
    """Water stress should be diagnosed correctly even when weather data is completely unavailable."""
    polygon = {
        "type": "Polygon",
        "coordinates": [[[76.0, 19.0], [76.1, 19.0], [76.1, 19.1], [76.0, 19.1], [76.0, 19.0]]],
    }
    state = {
        "polygon_geojson": polygon,
        "date_end": date(2026, 9, 15),
        "vision_classification": "water_stress",
        "vision_confidence": 0.82,
        "crop_type": "cotton",
    }
    with patch("agents.reasoning_agent.get_recent_weather", return_value={"total_rainfall_mm": None, "days": 14, "source": "unavailable"}):
        out = reasoning_agent(state)
        assert out["diagnosis"] == "Moisture Deficit Stress Pattern"
        assert out["risk_level"] == "high"
        assert "records were unavailable" in out["diagnosis_reasoning"]
