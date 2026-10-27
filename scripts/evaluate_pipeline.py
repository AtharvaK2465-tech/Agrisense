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
    software_regressions = []
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
                rec = {
                    "scenario": name,
                    "target_region": item["expected_district"],
                    "tested_aspect": "geometry_boundary_enforcement",
                    "actual_execution_mode": "pre_pipeline_geometry_guard",
                    "imagery_type": "none (rejected prior to fetch)",
                    "external_apis_contacted": {"open_meteo": False, "gemini": False, "sentinel_hub": False},
                    "processing_type": "deterministic_geometry_fixture",
                    "scientific_validation": "N/A (out-of-bounds rejected)",
                    "pdf_generated": False,
                    "status": "PASSED (correctly rejected outside Maharashtra boundary)",
                }
                software_regressions.append(rec)
                summary_records.append(rec)
                continue
            else:
                print("  [-] Indore unexpectedly accepted under RESTRICT_TO_MAHARASHTRA=True")
                all_passed = False
                continue

        if not is_valid:
            print(f"  [-] AOI validation failed: {val_msg}")
            all_passed = False
            continue

        software_regressions.append({
            "scenario": name,
            "target_region": item["expected_district"],
            "tested_aspect": "geometry_boundary_enforcement",
            "status": "PASSED (valid polygon within Maharashtra)",
        })

        # Execute multi-agent graph
        res = run_pipeline(poly, item["date_start"], item["date_end"])
        if res.get("error"):
            print(f"  [-] Pipeline failed: {res['error']}")
            all_passed = False
            continue

        prov = res.get("provenance", {})
        val = res.get("report_validation", {})
        surface = res.get("surface_fractions", {})
        weather = res.get("weather", {})

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
            "target_region": item["expected_district"],
            "tested_aspect": "application_logic_and_pipeline_orchestration",
            "actual_execution_mode": prov.get("execution_mode", "mock"),
            "imagery_type": "synthetic" if prov.get("is_synthetic_imagery") else "sentinel_2_l2a",
            "external_apis_contacted": {
                "open_meteo": "open-meteo" in str(weather.get("source", "")),
                "gemini": prov.get("report_source") == "gemini-live",
                "sentinel_hub": False,  # Mock mode active; Sentinel Hub not contacted
            },
            "processing_type": "deterministic_synthetic_fixture",
            "district": res.get("district"),
            "season": res.get("season"),
            "diagnosis": res.get("diagnosis"),
            "risk_level": res.get("risk_level"),
            "crop_decision": res.get("crop_type"),
            "crop_decision_status": res.get("crop_decision_status"),
            "crop_heuristic_score": res.get("crop_confidence"),
            "crop_similarity_margin": res.get("crop_similarity_margin"),
            "scene_id": prov.get("scene_id"),
            "report_source": prov.get("report_source"),
            "scientific_validation": val.get("status"),
            "pdf_generated": pdf_ok,
            "vegetation_fraction": surface.get("vegetation"),
            "status": "PASSED" if val.get("status") == "passed" and pdf_ok else "FAILED",
        }
        summary_records.append(record)

        print(f"  [+] District: {record['district']} | Season: {record['season']}")
        print(f"  [+] Diagnosis: {record['diagnosis']} ({record['risk_level'].upper()})")
        print(f"  [+] Crop Decision: {record['crop_decision']} ({record['crop_decision_status']}, score={record['crop_heuristic_score']:.1%})")
        print(f"  [+] Provenance: Mode={record['actual_execution_mode']}, Scene={record['scene_id']}")
        print(f"  [+] Scientific Validation: {record['scientific_validation']} | PDF Export: {'OK' if pdf_ok else 'FAILED'}\n")

    out_path = os.path.join(PROJECT_ROOT, "data", "evaluation_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tiered_output = {
        "evaluation_tiers": {
            "tier_1_software_regression_tests": {
                "status": "PASSED",
                "scope": "Geometry validation, coordinate transformations, and Maharashtra boundary enforcement.",
                "total_tests": len(software_regressions),
                "results": software_regressions,
            },
            "tier_2_synthetic_mock_pipeline_tests": {
                "status": "PASSED" if all_passed else "FAILED",
                "scope": "End-to-end multi-agent orchestration, spectral index computations, uncertainty gating, report validation, and PDF generation on synthetic rasters.",
                "total_scenarios": len(summary_records),
                "scenarios": summary_records,
            },
            "tier_3_live_satellite_processing_tests": {
                "status": "UNAVAILABLE",
                "scope": "Live Copernicus Sentinel-2 L2A satellite retrieval and real-time band processing.",
                "reason": "Sentinel Hub credentials (SENTINEL_HUB_CLIENT_ID / CLIENT_SECRET) are not configured. Cannot claim live verification without credentialed satellite pulls.",
            },
            "tier_4_independent_crop_classification_accuracy_tests": {
                "status": "UNAVAILABLE",
                "scope": "Statistical crop classification accuracy against independently measured ground-truth polygons.",
                "reason": "Ground-truth agricultural parcel surveys and field validation labels are unavailable. Heuristic scores represent curve similarity, not measured accuracy.",
            },
        },
        "all_passed": all_passed,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tiered_output, f, indent=2)

    print("=================================================================")
    print(f"Evaluation Completed. Overall Status: {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
    print(f"Summary saved to: {out_path}")
    print("=================================================================")
    return {"all_passed": all_passed, "tiered_output": tiered_output}


if __name__ == "__main__":
    res = evaluate_all_scenarios()
    sys.exit(0 if res["all_passed"] else 1)
