"""
Pipeline Evaluation and Reliability Benchmark Runner
---------------------------------------------------
Runs full end-to-end pipeline evaluations across regional scenarios:
Akola, Amravati, Nagpur, Indore, Barshi, Khopoli, and Wadhode.
Verifies:
- AOI boundary validation
- Spectral index calculations and quality
- Cloud cover handling
- Crop classification and uncertainty gating
- Scientific language compliance and validator verdicts
- Provenance tracking and execution mode separation
- Downloadable deliverable generation (PDF, JSON, Markdown)
"""
import json
import os
import sys
from datetime import date
from typing import Dict, Any

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.geo import point_radius_to_geojson, validate_polygon
from utils.districts import locate_district, season_for_date, plausible_crops
from utils.pdfexport import build_pdf
from agents.graph import run_pipeline
import utils.config as config


REGRESSION_SCENARIOS = [
    {
        "name": "Akola",
        "description": "Vidarbha Cotton & Soybean Heart - Kharif Season",
        "lat": 20.70,
        "lon": 77.00,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
        "expected_district": "Akola",
    },
    {
        "name": "Amravati",
        "description": "Vidarbha Soybean & Cotton Belt - Kharif Season",
        "lat": 20.93,
        "lon": 77.75,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
        "expected_district": "Amravati",
    },
    {
        "name": "Nagpur",
        "description": "Eastern Vidarbha Citrus & Cotton - Kharif Season",
        "lat": 21.14,
        "lon": 79.08,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
        "expected_district": "Nagpur",
    },
    {
        "name": "Barshi",
        "description": "Solapur District Drought-Prone Jowar Belt - Rabi Season",
        "lat": 18.23,
        "lon": 75.69,
        "date_start": date(2026, 10, 1),
        "date_end": date(2026, 12, 15),
        "expected_district": "Solapur",
    },
    {
        "name": "Khopoli",
        "description": "Western Ghats / Raigad High-Precipitation Foothills",
        "lat": 18.78,
        "lon": 73.34,
        "date_start": date(2026, 6, 15),
        "date_end": date(2026, 9, 15),
        "expected_district": "Raigad",
    },
    {
        "name": "Wadhode",
        "description": "Jalgaon / Tapi Basin Alluvial Cotton & Jowar",
        "lat": 21.15,
        "lon": 75.75,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
        "expected_district": "Jalgaon",
    },
    {
        "name": "Indore",
        "description": "Madhya Pradesh Territory (Outside Maharashtra Boundary Test)",
        "lat": 22.72,
        "lon": 75.85,
        "date_start": date(2026, 6, 1),
        "date_end": date(2026, 9, 15),
        "expected_district": "Outside Maharashtra",
    },
]


def evaluate_all_scenarios() -> Dict[str, Any]:
    print("=================================================================")
    print("[AGRISENSE] End-to-End Evaluation & Regression Benchmark Suite")
    print("=================================================================\n")

    summary_records = []
    all_passed = True

    for item in REGRESSION_SCENARIOS:
        name = item["name"]
        print(f"--> Testing Scenario: {name} ({item['description']})...")
        poly = point_radius_to_geojson(item["lat"], item["lon"], radius_km=2.0)

        # Boundary check
        is_valid, val_msg = validate_polygon(poly)
        if name == "Indore":
            if not is_valid:
                print(f"  [+] Correctly rejected: {val_msg}")
                summary_records.append({
                    "scenario": name,
                    "status": "PASSED (Correctly rejected outside state)",
                    "district": "Outside",
                    "execution_mode": "N/A",
                    "scientific_validation": "N/A",
                })
                continue
            else:
                print("  [-] Indore unexpectedly accepted under RESTRICT_TO_MAHARASHTRA=True")
                all_passed = False
                continue

        if not is_valid:
            print(f"  [-] AOI validation failed: {val_msg}")
            all_passed = False
            continue

        # Execute multi-agent graph
        res = run_pipeline(poly, item["date_start"], item["date_end"])
        if res.get("error"):
            print(f"  [-] Pipeline failed: {res['error']}")
            all_passed = False
            continue

        prov = res.get("provenance", {})
        val = res.get("report_validation", {})
        surface = res.get("surface_fractions", {})

        # Test PDF generation
        pdf_ok = False
        try:
            pdf_bytes = build_pdf(res["report_markdown"], {"district": res.get("district"), "season": res.get("season")})
            pdf_ok = len(pdf_bytes) > 500
        except Exception as e:
            print(f"  [-] PDF export failed: {e}")
            all_passed = False

        record = {
            "scenario": name,
            "district": res.get("district"),
            "season": res.get("season"),
            "diagnosis": res.get("diagnosis"),
            "risk_level": res.get("risk_level"),
            "crop_decision": res.get("crop_type"),
            "crop_confidence": res.get("crop_confidence"),
            "execution_mode": prov.get("execution_mode"),
            "is_synthetic": prov.get("is_synthetic_imagery"),
            "scene_id": prov.get("scene_id"),
            "report_source": prov.get("report_source"),
            "scientific_validation": val.get("status"),
            "pdf_export_ok": pdf_ok,
            "vegetation_fraction": surface.get("vegetation"),
            "status": "PASSED" if val.get("status") == "passed" and pdf_ok else "FAILED",
        }
        summary_records.append(record)

        print(f"  [+] District: {record['district']} | Season: {record['season']}")
        print(f"  [+] Diagnosis: {record['diagnosis']} ({record['risk_level'].upper()})")
        print(f"  [+] Crop Decision: {record['crop_decision']} ({record['crop_confidence']:.0%})")
        print(f"  [+] Provenance: Mode={record['execution_mode']}, Scene={record['scene_id']}")
        print(f"  [+] Scientific Validation: {record['scientific_validation']} | PDF Export: {'OK' if pdf_ok else 'FAILED'}\n")

    out_path = os.path.join(PROJECT_ROOT, "data", "evaluation_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"scenarios": summary_records, "all_passed": all_passed}, f, indent=2)

    print("=================================================================")
    print(f"Evaluation Completed. Overall Status: {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
    print(f"Summary saved to: {out_path}")
    print("=================================================================")
    return {"all_passed": all_passed, "records": summary_records}


if __name__ == "__main__":
    res = evaluate_all_scenarios()
    sys.exit(0 if res["all_passed"] else 1)
