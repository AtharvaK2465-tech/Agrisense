"""
Adversarial and compliance testing for Scientific Language Safeguards (commit 5b1b064).
Ensures that the ReportValidator rejects prescriptive directives, unsupported pathogen claims,
causality overreach, and ungrounded crop-stage certainty while accepting legitimate cautious
negative assertions.
"""
import pytest
from agents.report_validator import validate_report, sanitize_report_violations


def test_reject_prescriptive_pesticide_directive():
    bad_report = """## Crop Health Report
### Diagnosis
Vegetative stress observed in cotton parcel.
### Recommendation
Immediately spray 500ml chlorpyrifos insecticide across the affected acreage."""
    result = validate_report(bad_report, {})
    assert result["is_valid"] is False
    assert result["status"] == "rejected"
    assert any("chemical" in v.lower() or "directive" in v.lower() for v in result["violations"])


def test_reject_prescriptive_fertilizer_directive():
    bad_report = """## Crop Health Report
### Diagnosis
Canopy vigor depression detected.
### Recommendation
Apply 50 kg urea fertilizer per acre to correct nitrogen deficiency."""
    result = validate_report(bad_report, {})
    assert result["is_valid"] is False
    assert any("chemical" in v.lower() or "fertilizer" in v.lower() for v in result["violations"])


def test_reject_definitive_pathogen_assertion():
    bad_report = """## Crop Health Report
### Diagnosis
Satellite confirms the disease red rot pathogen in sugarcane.
### Recommendation
Inspect the parcel and consult an agronomist."""
    result = validate_report(bad_report, {})
    assert result["is_valid"] is False
    assert any("pathogen" in v.lower() for v in result["violations"])


def test_reject_weather_causality_overreach():
    bad_report = """## Crop Health Report
### Diagnosis
Heavy rain caused root rot and guaranteed drainage failure.
### Recommendation
Conduct field scouting."""
    result = validate_report(bad_report, {})
    assert result["is_valid"] is False
    assert any("causality" in v.lower() for v in result["violations"])


def test_reject_unverified_crop_stage_certainty():
    bad_report = """## Crop Health Report
### Diagnosis
Crop is confirmed at stage 3 peak growth.
### Recommendation
Field inspection recommended."""
    result = validate_report(bad_report, {})
    assert result["is_valid"] is False
    assert any("crop-stage" in v.lower() or "stage" in v.lower() for v in result["violations"])


def test_allow_legitimate_cautious_negative_statements():
    """Legitimate negative/cautious statements must NOT trigger false positive rejection."""
    good_report = """## Crop Health Report
### Observations
Spectral anomaly observed across the canopy. However, satellite indices cannot confirm
specific fungal pathogen or disease organism without physical laboratory testing.
Do not spray chemical pesticide without confirmed field diagnosis.
### Recommendation
Conduct in-field ground survey and consult local Krishi Vigyan Kendra extension officer."""
    result = validate_report(good_report, {})
    assert result["is_valid"] is True
    assert result["status"] == "passed"
    assert len(result["violations"]) == 0
    assert "no_prescriptive_chemical_directives" in result["passed_checks"]
    assert "no_unsupported_pathogen_claims" in result["passed_checks"]


def test_sanitize_report_violations_appends_disclaimer():
    raw_text = "Report with unverified assertions."
    violations = ["Directive found"]
    sanitized = sanitize_report_violations(raw_text, violations)
    assert raw_text in sanitized
    assert "Scientific Validation Notice" in sanitized
    assert "Do not apply chemical pesticides" in sanitized
