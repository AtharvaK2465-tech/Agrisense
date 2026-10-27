"""
Scientific Report Validator
---------------------------
Enforces rigorous scientific language rules (associated with commit 5b1b064):
  1. No prescriptive chemical / pesticide / fertilizer applications solely from satellite data.
  2. No definitive pathogen / disease assertions without ground-truth physical lab confirmation.
  3. No ungrounded causal claims (e.g., weather causing specific root pathology or proving crop failure).
  4. No unvalidated crop-stage certainty (satellite phenology provides estimates, not confirmed stages).
  5. Mandates recommendations for physical field verification and agronomic consultation.
  6. Permits legitimate cautious negative statements ("cannot confirm", "evidence is insufficient").
"""
import re
from typing import Dict, Any, List, Tuple


# Regex patterns that indicate unsupported claims or prescriptive directives
# (Matched only when NOT preceded by negative / cautious words like "cannot", "do not", "insufficient")
FORBIDDEN_PRESCRIPTIVE_DIRECTIVES = [
    r"\b(?:spray|apply|administer|dose)\s+(?:chemical|pesticide|insecticide|herbicide|fungicide|urea|fertilizer|monocrotophos|chlorpyrifos|glyphosate)\b",
    r"\b(?:apply|spray)\s+\d+\s*(?:kg|ml|liters?|gm)\b",
    r"\b(?:must|should|immediately)\s+(?:spray|apply\s+chemical)\b",
    r"\birrigate\s+(?:immediately\s+with|\d+\s*mm\s+of\s+water)\b",
]

FORBIDDEN_PATHOGEN_ASSERTIONS = [
    r"\b(?:definitely|confirmed|proven)\s+(?:infected|pathogen|fungus|fungal\s+infection|virus|bacterial\s+blight|red\s+rot\s+disease|bollworm\s+outbreak)\b",
    r"\bsatellite\s+(?:confirms|proves)\s+(?:the\s+presence\s+of|the\s+disease|pathogen)\b",
    r"\bproven\s+infestation\b",
]

FORBIDDEN_CAUSALITY_OVERREACH = [
    r"\b(?:rainfall|rain)\s+(?:proves|caused|guarantees)\s+(?:root\s+rot|drainage\s+failure|fungal\s+attack)\b",
    r"\b(?:low\s+ndvi|ndvi\s+drop)\s+(?:proves|confirms)\s+(?:crop\s+failure|drought\s+devastation|terminal\s+damage)\b",
    r"\bdrought\s+has\s+caused\s+complete\s+crop\s+loss\b",
]

FORBIDDEN_CROP_STAGE_CERTAINTY = [
    r"\bcrop\s+is\s+confirmed\s+at\s+stage\s+\d+\b",
    r"\bconfirmed\s+to\s+be\s+(?:unplanted|tilled|pre-emergence)\b",
    r"\bfield\s+is\s+definitely\s+(?:unplanted|fallow)\b",
]

# Safeguard check: legitimate negative hedges that neutralize false positive matches
CAUTIOUS_HEDGE_PATTERNS = [
    r"\bcannot\s+(?:confirm|establish|determine|be\s+used\s+to)\b",
    r"\bdoes\s+not\s+(?:confirm|prove|establish|substitute)\b",
    r"\binsufficient\s+evidence\b",
    r"\bdo\s+not\s+(?:spray|apply)\b",
    r"\bnot\s+recommended\s+without\s+field\s+testing\b",
    r"\bground\s+verification\s+required\b",
    r"\brequires?\s+physical\s+sampling\b",
]


def validate_report(report_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a generated report against scientific integrity rules.
    Returns:
        {
            "is_valid": bool,
            "status": "passed" | "rejected",
            "passed_checks": list[str],
            "violations": list[str],
            "warnings": list[str],
            "details": dict,
        }
    """
    violations: List[str] = []
    warnings: List[str] = []
    passed_checks: List[str] = []

    text_lower = report_text.lower()

    # Split report into sentences for context-aware check
    sentences = re.split(r"[.!?\n]+", text_lower)

    # 1. Check Prescriptive Chemical Directives
    directive_found = False
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Check if sentence has a cautious negative hedge
        if any(re.search(hedge, sent) for hedge in CAUTIOUS_HEDGE_PATTERNS):
            continue
        for pat in FORBIDDEN_PRESCRIPTIVE_DIRECTIVES:
            if re.search(pat, sent):
                violations.append(f"Prescriptive chemical/fertilizer directive found: '{sent[:80]}...'")
                directive_found = True
                break
        if directive_found:
            break
    if not directive_found:
        passed_checks.append("no_prescriptive_chemical_directives")

    # 2. Check Definitive Pathogen / Disease Assertions
    pathogen_found = False
    for sent in sentences:
        sent = sent.strip()
        if not sent or any(re.search(hedge, sent) for hedge in CAUTIOUS_HEDGE_PATTERNS):
            continue
        for pat in FORBIDDEN_PATHOGEN_ASSERTIONS:
            if re.search(pat, sent):
                violations.append(f"Unsubstantiated pathogen certainty claim found: '{sent[:80]}...'")
                pathogen_found = True
                break
        if pathogen_found:
            break
    if not pathogen_found:
        passed_checks.append("no_unsupported_pathogen_claims")

    # 3. Check Weather & NDVI Causality Overreach
    causality_found = False
    for sent in sentences:
        sent = sent.strip()
        if not sent or any(re.search(hedge, sent) for hedge in CAUTIOUS_HEDGE_PATTERNS):
            continue
        for pat in FORBIDDEN_CAUSALITY_OVERREACH:
            if re.search(pat, sent):
                violations.append(f"Unsubstantiated causality claim found: '{sent[:80]}...'")
                causality_found = True
                break
        if causality_found:
            break
    if not causality_found:
        passed_checks.append("no_causality_overreach")

    # 4. Check Crop Stage Certainty
    stage_found = False
    for sent in sentences:
        sent = sent.strip()
        if not sent or any(re.search(hedge, sent) for hedge in CAUTIOUS_HEDGE_PATTERNS):
            continue
        for pat in FORBIDDEN_CROP_STAGE_CERTAINTY:
            if re.search(pat, sent):
                violations.append(f"Unverified crop-stage certainty claim found: '{sent[:80]}...'")
                stage_found = True
                break
        if stage_found:
            break
    if not stage_found:
        passed_checks.append("no_unvalidated_crop_stage_certainty")

    # 5. Check Ground Verification Advice
    verification_keywords = [
        "field inspection", "ground survey", "field verification", "scouting",
        "consult", "agronomist", "extension", "soil test", "in-field", "non-agricultural",
    ]
    has_verification_rec = any(kw in text_lower for kw in verification_keywords)
    if has_verification_rec:
        passed_checks.append("includes_field_verification_recommendation")
    else:
        warnings.append("Report lacks explicit recommendation for field verification or expert consultation.")

    is_valid = len(violations) == 0

    return {
        "is_valid": is_valid,
        "status": "passed" if is_valid else "rejected",
        "passed_checks": passed_checks,
        "violations": violations,
        "warnings": warnings,
    }


def sanitize_report_violations(report_text: str, violations: List[str]) -> str:
    """Appends an explicit scientific disclaimer when violations or warnings are flagged."""
    disclaimer = (
        "\n\n> [!CAUTION]\n"
        "> **Scientific Validation Notice:** Satellite spectral observations detect vegetation vigor anomalies, "
        "not specific biological pathogens, chemical deficiencies, or drought causality. Do not apply chemical "
        "pesticides or alter crop management without direct physical field scouting and consultation with a local agronomist."
    )
    return report_text + disclaimer
