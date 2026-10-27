"""
Streamlit entry point. Draw one or more polygons on the map over
Maharashtra, pick a date range, and run the multi-agent pipeline against
each area: Geo -> Input(retry loop) -> Crop-Type -> Vision -> Reasoning
-> Report.

Run with:  streamlit run app.py
"""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape

from utils.config import (
    MAHARASHTRA_BBOX,
    SENTINEL_HUB_MOCK,
    GEMINI_MOCK,
    is_gemini_mock,
    get_gemini_api_key,
    set_gemini_api_key,
)
from utils.crops import CROP_COLORS, CROP_DISPLAY_NAMES
from utils.colormap import ndvi_to_rgba, confidence_to_rgba
from utils.pdfexport import build_pdf
from utils.history import load_history, save_run
from agents.graph import run_pipeline

st.set_page_config(page_title="Crop Health Monitor — Maharashtra", layout="wide")
st.title("🌾 Multi-Agent Crop Health Monitoring — Maharashtra")

gemini_in_mock = is_gemini_mock()
if SENTINEL_HUB_MOCK or gemini_in_mock:
    modes = []
    if SENTINEL_HUB_MOCK:
        modes.append("Sentinel Hub")
    if gemini_in_mock:
        modes.append("Gemini")
    st.info(
        f"Running in **mock mode** for {', '.join(modes)} — add real API keys in the sidebar or `.env` to switch to live data. "
        "The full pipeline still runs end-to-end with realistic synthetic data.",
        icon="🧪",
    )


METHODOLOGY_MD = """
**Indices.** NDVI = (NIR−Red)/(NIR+Red). NDMI/NDRE approximate moisture and
red-edge stress; live mode uses real Sentinel-2 SWIR/red-edge bands, mock
mode approximates them from RGB+NIR (see `agents/geo_agent.py`).

**3-Stage Evidence-Based Crop Pipeline.**
1. *Agricultural Eligibility:* Distinguishes genuine agricultural terrain from non-agricultural land
   (urban residential, college campus grounds like VIT, industrial estates, roads, water bodies).
   *Scientific safeguard:* Low NDVI alone (e.g. from early emergence, recent harvest, or drought)
   does *not* cause agricultural fields to be flagged as non-agricultural.
2. *Crop Phenology Matching:* Multi-date NDVI series across 4 growth stages are compared against
   reference signatures using Root-Mean-Square (RMS) spectral distance (not black-box ML).
   Normalized similarity weights are computed via inverse-distance weighting.
3. *Confidence & Uncertainty Gating:* Explicitly evaluates whether the best-matching crop
   has a sufficient margin over the second-best candidate and satisfies seasonal/district plausibility.
   If evidence is ambiguous or weak, the system explicitly outputs **"Agricultural Area (Crop Uncertain)"**
   rather than forcing an unvalidated prediction.

**Confidence.** The crop-match heatmap colors each pixel by how close its
NDVI is to the predicted crop's expected value at this stage (a ±0.15 NDVI
tolerance band) — this is a per-pixel *consistency* score, not a
calibrated statistical probability. The stress diagnosis's own
confidence is a transparent heuristic over mean NDVI/NDMI
(`agents/vision_agent.py`).

**Historical baseline.** The "3-year average" comparison is a seeded
synthetic reconstruction for demo purposes — wire real multi-year
Sentinel-2 archive pulls for production use.

**Known caveat.** All imagery and multi-date NDVI series are synthetic
(mock mode) unless real Sentinel Hub credentials are set in `.env` — the
classification/reasoning logic itself is identical in mock and live mode.
"""


def _legend_html() -> str:
    rows = "".join(
        f'<div style="margin-top:2px;"><span style="display:inline-block;width:12px;'
        f'height:12px;background:{color};margin-right:6px;border:1px solid #0003;"></span>'
        f'{CROP_DISPLAY_NAMES[crop]}</div>'
        for crop, color in CROP_COLORS.items()
    )
    return f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: white; padding: 8px 12px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.35); font-size: 12px; line-height: 1.3;">
      <b>Predicted crop</b>{rows}
    </div>
    """


def _polygon_bounds(geojson: dict):
    minx, miny, maxx, maxy = shape(geojson).bounds
    return [[miny, minx], [maxy, maxx]]


def _to_uint8_rgba(rgba: np.ndarray) -> np.ndarray:
    return (np.clip(rgba, 0, 1) * 255).astype(np.uint8)


def _build_result_map(polygon_geojson: dict, ndvi: np.ndarray, confidence_map: np.ndarray, crop_type: str) -> folium.Map:
    bounds = _polygon_bounds(polygon_geojson)
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    m = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=13)
    m.fit_bounds(bounds)

    folium.raster_layers.ImageOverlay(
        image=_to_uint8_rgba(ndvi_to_rgba(ndvi)),
        bounds=bounds, opacity=0.65, name="NDVI (red→yellow→green)",
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        image=_to_uint8_rgba(confidence_to_rgba(confidence_map)),
        bounds=bounds, opacity=0.45, name="Crop-match confidence", show=False,
    ).add_to(m)

    color = CROP_COLORS.get(crop_type, "#999999")
    folium.GeoJson(
        {"type": "Feature", "geometry": polygon_geojson, "properties": {}},
        style_function=lambda f, c=color: {"fillColor": c, "color": c, "weight": 3, "fillOpacity": 0.12},
        name=f"Predicted crop: {CROP_DISPLAY_NAMES.get(crop_type, crop_type)}",
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html()))
    return m


def _render_area_result(label: str, result: dict, index: int):
    if result.get("error"):
        st.error(f"{label}: {result['error']}")
        return

    st.success(result.get("validation_notes", "Pipeline completed."))

    # Provenance Banner
    prov = result.get("provenance", {})
    exec_mode = prov.get("execution_mode", "mock")
    if exec_mode == "live":
        st.success(
            f"🛰️ **Data Source: Live Satellite Imagery** | Scene: `{prov.get('scene_id', 'S2-L2A')}` | "
            f"Acquisition: {prov.get('acquisition_date', 'N/A')} | Provider: {prov.get('provider', 'Copernicus')}",
            icon="🟢",
        )
    elif exec_mode == "live_fallback_to_mock":
        st.warning(
            f"⚠️ **Data Source: Synthetic Fallback** | Live satellite retrieval failed; reverted to synthetic demonstration raster. "
            f"Scene ID: `{prov.get('scene_id', 'SYNTH')}`. THIS IS NOT LIVE SATELLITE DATA.",
            icon="⚠️",
        )
    else:
        st.info(
            f"🧪 **Data Source: Synthetic Raster (Mock Mode)** | Seeded demonstration raster: `{prov.get('scene_id', 'SYNTH')}` | "
            f"Tile: {prov.get('mgrs_tile', 'MOCK-43QEA')}",
            icon="🧪",
        )

    # Core metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observed Diagnosis", result["diagnosis"].replace("_", " ").title())
    c2.metric("Assessed Risk", result["risk_level"].upper())
    c3.metric("Vision Consistency", f"{result['vision_confidence']:.0%}")
    c4.metric("Identified Crop / Ground", CROP_DISPLAY_NAMES.get(result.get("crop_type", ""), "—"))
    c5.metric("Decision Confidence", f"{result.get('crop_confidence', 0):.0%}")
    
    aoi_info = prov.get("aoi", {})
    st.caption(
        f"District: **{result.get('district', 'Unknown')}** · Season: **{result.get('season', 'Unknown').title()}** · "
        f"AOI Area: **{aoi_info.get('area_km2', 0):,.2f} km²** ({aoi_info.get('area_hectares', 0):,.1f} ha) · "
        f"Cloud Cover: **{result.get('cloud_cover_pct', 0):.1f}%**"
    )

    if result.get("crop_decision_reason"):
        if result.get("crop_decision_status") == "no_crop":
            st.info(f"🏢 **Land Assessment:** {result['crop_decision_reason']}")
        elif result.get("crop_decision_status") == "uncertain":
            st.warning(f"⚠️ **Crop Uncertainty:** {result['crop_decision_reason']}")
        else:
            st.info(f"🌱 **Crop Decision Evidence:** {result['crop_decision_reason']}")

    # Spectral Indices & Surface Breakdown
    st.subheader("Spectral Indices & Surface Fractions")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        mean_ndvi = float(np.mean(result["ndvi"])) if result.get("ndvi") is not None else 0.0
        st.metric("Mean NDVI (Vigor)", f"{mean_ndvi:.3f}")
    with s2:
        if result.get("ndmi") is not None:
            st.metric("Mean NDMI (Moisture)", f"{float(np.mean(result['ndmi'])):.3f}")
        else:
            ndmi_reason = prov.get("indices_quality", {}).get("ndmi", {}).get("reason", "SWIR band unavailable")
            st.metric("Mean NDMI", "Unavailable", help=ndmi_reason)
    with s3:
        if result.get("ndre") is not None:
            st.metric("Mean NDRE (RedEdge)", f"{float(np.mean(result['ndre'])):.3f}")
        else:
            ndre_reason = prov.get("indices_quality", {}).get("ndre", {}).get("reason", "RedEdge band unavailable")
            st.metric("Mean NDRE", "Unavailable", help=ndre_reason)
    with s4:
        st.metric("Valid Pixel Coverage", f"{result.get('valid_pixels_pct', 100.0):.1f}%")

    surface = result.get("surface_fractions", {})
    st.caption(
        f"🌾 **Active Canopy:** {surface.get('vegetation', 0.0):.1%} | "
        f"🟤 **Bare Soil / Stubble:** {surface.get('bare_surface', 0.0):.1%} | "
        f"💧 **Water Surface:** {surface.get('water_surface', 0.0):.1%}"
    )

    st.subheader("Map — NDVI + crop-type overlay")
    result_map = _build_result_map(
        result["polygon_geojson"], result["ndvi"], result["crop_confidence_map"], result.get("crop_type", "")
    )
    st_folium(result_map, height=420, width=None, key=f"result_map_{index}")

    col_ts, col_ba = st.columns(2)
    with col_ts:
        st.subheader("NDVI across growing season stages")
        b_status = result.get("baseline_status", "unavailable")
        if b_status == "unavailable" or result.get("baseline_ndvi") is None:
            df = pd.DataFrame(
                {"Observed NDVI": result["ndvi_timeseries"]},
                index=result["phenology_stage_labels"],
            )
            st.line_chart(df)
            st.caption(f"ℹ️ 3-year baseline is unavailable ({result.get('baseline_reason', 'multi-year satellite query not executed')}).")
        else:
            df = pd.DataFrame(
                {
                    "NDVI": result["ndvi_timeseries"],
                    "3-yr baseline": [result["baseline_ndvi"]] * len(result["ndvi_timeseries"]),
                },
                index=result["phenology_stage_labels"],
            )
            st.line_chart(df)
            if b_status == "synthetic_simulation":
                st.caption("ℹ️ Baseline is a simulated 3-year average for demonstration.")
            else:
                st.caption(result.get("crop_agent_notes", ""))

    with col_ba:
        st.subheader("Before / after — vs. 3-yr baseline")
        if result.get("baseline_ndvi") is not None and result.get("ndvi_anomaly") is not None:
            baseline_ndvi_arr = np.clip(result["ndvi"] - result["ndvi_anomaly"], -1, 1)
            b1, b2 = st.columns(2)
            b1.image(_to_uint8_rgba(ndvi_to_rgba(baseline_ndvi_arr)), caption="3-yr seasonal baseline", use_container_width=True)
            b2.image(_to_uint8_rgba(ndvi_to_rgba(result["ndvi"])), caption="Current", use_container_width=True)
            anomaly = result["ndvi_anomaly"]
            st.caption(f"Anomaly: {anomaly:+.2f} NDVI vs. baseline "
                       f"({'unusually stressed' if anomaly < -0.08 else 'unusually vigorous' if anomaly > 0.08 else 'within normal range'}).")
        else:
            st.info("Historical baseline imagery comparison is unavailable (requires multi-year satellite catalog retrieval).")

    st.subheader("Validation — known issues in this district/crop/season")
    matches = result.get("advisory_matches", [])
    if matches:
        st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
        st.caption("⚠️ Demo reference data — replace with a real advisory source for production use.")
    else:
        st.caption("No matching demo advisory records for this district/crop/season.")

    with st.expander("How this was calculated (methodology)"):
        st.markdown(METHODOLOGY_MD)

    st.subheader("Report")
    val_status = result.get("report_validation", {})
    if val_status.get("is_valid"):
        st.success("🛡️ **Scientific Integrity Verified:** No prescriptive chemical sprays, unverified pathogen assertions, or causality overreach.")
    else:
        st.warning(f"⚠️ **Scientific Validation Flag:** {val_status.get('violations')}")

    st.markdown(result["report_markdown"])

    # Downloadable deliverables: PDF, Markdown, and JSON
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        pdf_bytes = None
        try:
            pdf_bytes = build_pdf(
                result["report_markdown"],
                {
                    "district": result.get("district", "-"),
                    "season": result.get("season", "-"),
                    "crop_type": CROP_DISPLAY_NAMES.get(result.get("crop_type", ""), "-"),
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            )
        except Exception as e:
            st.warning(f"Couldn't generate PDF ({e}).")

        if pdf_bytes is not None:
            st.download_button(
                f"⬇️ Download PDF Report", data=pdf_bytes,
                file_name=f"crop_health_report_{label.replace(' ', '_').lower()}.pdf",
                mime="application/pdf", key=f"pdf_{index}", use_container_width=True,
            )

    with col_d2:
        st.download_button(
            "⬇️ Download Markdown Report",
            data=result["report_markdown"],
            file_name=f"crop_health_report_{label.replace(' ', '_').lower()}.md",
            mime="text/markdown", key=f"md_{index}", use_container_width=True,
        )

    with col_d3:
        # Prepare clean JSON export without numpy arrays
        clean_export = {
            "label": label,
            "generated_at": datetime.now().isoformat(),
            "provenance": prov,
            "diagnosis": result.get("diagnosis"),
            "risk_level": result.get("risk_level"),
            "crop_type": result.get("crop_type"),
            "crop_confidence": result.get("crop_confidence"),
            "crop_scores": result.get("crop_scores"),
            "crop_evidence": result.get("crop_evidence"),
            "district": result.get("district"),
            "season": result.get("season"),
            "indices_quality": result.get("indices_quality"),
            "surface_fractions": result.get("surface_fractions"),
            "weather": result.get("weather"),
            "report_validation": result.get("report_validation"),
        }
        import json
        st.download_button(
            "⬇️ Download JSON Provenance & Data",
            data=json.dumps(clean_export, indent=2),
            file_name=f"crop_health_provenance_{label.replace(' ', '_').lower()}.json",
            mime="application/json", key=f"json_{index}", use_container_width=True,
        )


with st.sidebar:
    st.subheader("⚙️ API Configuration")
    current_gemini_key = get_gemini_api_key()
    if current_gemini_key:
        st.success("🟢 Gemini API: **Live Mode**", icon="✨")
    else:
        st.info("🧪 Gemini API: **Mock Mode**", icon="🧪")

    with st.expander("Configure Gemini API Key", expanded=not bool(current_gemini_key)):
        new_key = st.text_input(
            "Gemini API Key",
            value=current_gemini_key,
            type="password",
            placeholder="AIzaSy...",
            help="Get a free key from Google AI Studio: https://aistudio.google.com/apikey",
        )
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            if st.button("Save Key", type="primary", use_container_width=True):
                set_gemini_api_key(new_key.strip())
                st.success("API Key saved to .env!")
                st.rerun()
        with col_k2:
            if st.button("Clear Key", use_container_width=True):
                set_gemini_api_key("")
                st.rerun()

    st.divider()

    st.subheader("📜 Run history")
    history = load_history()
    if not history:
        st.caption("No runs saved yet. Run an analysis to see it here.")
    else:
        for entry in history[:15]:
            mode_badge = "🟢 Live" if entry.get("execution_mode") == "live" else "🧪 Mock"
            with st.expander(f"{entry.get('label', 'Area')} [{mode_badge}] — {entry.get('saved_at', '')[:16]}"):
                st.write(f"**Data Source:** {entry.get('execution_mode', 'mock').upper()} ({'Synthetic' if entry.get('is_synthetic', True) else 'Live S2'})")
                if entry.get("scene_id"):
                    st.write(f"**Scene ID:** `{entry.get('scene_id')}`")
                st.write(f"**Diagnosis:** {entry.get('diagnosis', '—')}")
                st.write(f"**Risk:** {entry.get('risk_level', '—')}")
                st.write(f"**Crop:** {entry.get('crop_type', '—')}")
                st.write(f"**District/season:** {entry.get('district', '—')} / {entry.get('season', '—')}")

col_map, col_controls = st.columns([2, 1])

with col_controls:
    st.subheader("1. Select date range")
    end_date = st.date_input("End date", value=date.today() - timedelta(days=3))
    start_date = st.date_input("Start date", value=end_date - timedelta(days=14))
    run_button = st.button("Run analysis on drawn area(s)", type="primary")

with col_map:
    st.subheader("2. Draw your area(s) of interest")
    st.caption("Draw one or more polygons/rectangles — each is analyzed and reported separately.")
    minx, miny, maxx, maxy = MAHARASHTRA_BBOX
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    tile_choice = st.radio(
        "Basemap",
        ["Satellite (Esri)", "Satellite + labels (Esri hybrid)", "Streets (OSM)"],
        horizontal=True,
        help=(
            "Esri's satellite imagery isn't updated on the same schedule "
            "everywhere — rural Maharashtra can lag behind what you'd see "
            "on Google Maps. If buildings/roads look outdated, switch to "
            "Streets to confirm your area against current OpenStreetMap "
            "data, then switch back to draw."
        ),
    )

    m = folium.Map(location=center, zoom_start=6, tiles=None)

    if tile_choice == "Streets (OSM)":
        folium.TileLayer("OpenStreetMap", name="Streets", max_zoom=19).add_to(m)
    else:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics", name="Satellite", max_zoom=19,
        ).add_to(m)
        if tile_choice == "Satellite + labels (Esri hybrid)":
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", name="Labels", overlay=True, control=False, max_zoom=19,
            ).add_to(m)

    Draw(
        export=False,
        draw_options={"polygon": True, "rectangle": True, "circle": False,
                      "circlemarker": False, "marker": False, "polyline": False},
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    map_data = st_folium(m, height=520, width=None, key="area_map")

st.divider()

polygons = []
if map_data and map_data.get("all_drawings"):
    polygons = [f["geometry"] for f in map_data["all_drawings"] if f.get("geometry")]

with st.expander("How this is calculated (methodology)"):
    st.markdown(METHODOLOGY_MD)

if run_button:
    if not polygons:
        st.error("Draw at least one polygon or rectangle on the map first.")
    else:
        results = []
        with st.spinner(f"Running Geo → Input → Crop-Type → Vision → Reasoning → Report agents on {len(polygons)} area(s)..."):
            for i, poly in enumerate(polygons, start=1):
                result = run_pipeline(poly, start_date, end_date)
                results.append((f"Area {i}", result))
                if not result.get("error"):
                    save_run({
                        "label": f"Area {i}",
                        "diagnosis": result["diagnosis"],
                        "risk_level": result["risk_level"],
                        "crop_type": result.get("crop_type"),
                        "crop_confidence": result.get("crop_confidence"),
                        "district": result.get("district"),
                        "season": result.get("season"),
                        "execution_mode": result.get("provenance", {}).get("execution_mode", "mock"),
                        "is_synthetic": result.get("provenance", {}).get("is_synthetic_imagery", True),
                        "scene_id": result.get("provenance", {}).get("scene_id"),
                        "report_source": result.get("provenance", {}).get("report_source"),
                        "validation_status": result.get("report_validation", {}).get("status"),
                    })
        st.session_state["last_results"] = results

if "last_results" in st.session_state:
    results = st.session_state["last_results"]

    if len(results) > 1:
        st.subheader("Compare areas")
        rows = []
        for label, r in results:
            if r.get("error"):
                continue
            rows.append({
                "Area": label, "Diagnosis": r["diagnosis"].replace("_", " ").title(),
                "Risk": r["risk_level"].upper(), "Crop": CROP_DISPLAY_NAMES.get(r.get("crop_type", ""), "—"),
                "Crop confidence": f"{r.get('crop_confidence', 0):.0%}",
                "District": r.get("district", "—"), "NDVI anomaly": f"{r.get('ndvi_anomaly', 0):+.2f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.divider()

    tabs = st.tabs([label for label, _ in results])
    for i, (tab, (label, result)) in enumerate(zip(tabs, results)):
        with tab:
            _render_area_result(label, result, i)

st.caption(
    "Multi-agent pipeline: Geo/Area Agent → Input/Validation Agent "
    "(with cloud-cover retry loop) → Crop-Type Agent → Vision Agent → "
    "Reasoning Agent → Report Agent."
)
