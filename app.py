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

from utils.config import MAHARASHTRA_BBOX, SENTINEL_HUB_MOCK, GEMINI_MOCK
from utils.crops import CROP_COLORS, CROP_DISPLAY_NAMES
from utils.colormap import ndvi_to_rgba, confidence_to_rgba
from utils.pdfexport import build_pdf
from utils.history import load_history, save_run
from agents.graph import run_pipeline

st.set_page_config(page_title="Crop Health Monitor — Maharashtra", layout="wide")
st.title("🌾 Multi-Agent Crop Health Monitoring — Maharashtra")

if SENTINEL_HUB_MOCK or GEMINI_MOCK:
    st.info(
        "Running in **mock mode** for "
        + ", ".join(
            [s for s, flag in [("Sentinel Hub", SENTINEL_HUB_MOCK), ("Gemini", GEMINI_MOCK)] if flag]
        )
        + " — add real API keys to `.env` to switch to live data. "
        "The full pipeline still runs end-to-end with realistic synthetic data.",
        icon="🧪",
    )

METHODOLOGY_MD = """
**Indices.** NDVI = (NIR−Red)/(NIR+Red). NDMI/NDRE approximate moisture and
red-edge stress; live mode uses real Sentinel-2 SWIR/red-edge bands, mock
mode approximates them from RGB+NIR (see `agents/geo_agent.py`).

**Crop-type classification.** NDVI is sampled at 4 phenology stages across
the growing season and compared (root-mean-square distance) against
reference signatures for cotton, soybean, sugarcane, jowar and tur — a
training-data-free technique based on real curve-shape differences between
crops, not a black box. The result is then cross-checked against the
district's Kharif/Rabi crop calendar and re-ranked if implausible.
See `utils/crops.py` and `utils/districts.py` for the exact reference
values, colors and calendar rules.

**Confidence.** The crop-match heatmap colors each pixel by how close its
NDVI is to the predicted crop's expected value at this stage (a ±0.15 NDVI
tolerance band) — this is a per-pixel *consistency* score, not a
probability from a trained classifier. The stress diagnosis's own
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

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Diagnosis", result["diagnosis"].replace("_", " ").title())
    c2.metric("Risk level", result["risk_level"].upper())
    c3.metric("Vision confidence", f"{result['vision_confidence']:.0%}")
    c4.metric("Crop type", CROP_DISPLAY_NAMES.get(result.get("crop_type", ""), "—"))
    c5.metric("Crop confidence", f"{result.get('crop_confidence', 0):.0%}")
    st.caption(f"District: **{result.get('district', 'Unknown')}** · Season: **{result.get('season', 'Unknown').title()}**")

    st.subheader("Map — NDVI + crop-type overlay")
    result_map = _build_result_map(
        result["polygon_geojson"], result["ndvi"], result["crop_confidence_map"], result.get("crop_type", "")
    )
    st_folium(result_map, height=420, width=None, key=f"result_map_{index}")

    col_ts, col_ba = st.columns(2)
    with col_ts:
        st.subheader("NDVI across the growing season")
        df = pd.DataFrame(
            {
                "NDVI": result["ndvi_timeseries"],
                "3-yr baseline": [result["baseline_ndvi"]] * len(result["ndvi_timeseries"]),
            },
            index=result["phenology_stage_labels"],
        )
        st.line_chart(df)
        st.caption(result.get("crop_agent_notes", ""))

    with col_ba:
        st.subheader("Before / after — vs. 3-yr baseline")
        baseline_ndvi_arr = np.clip(result["ndvi"] - result["ndvi_anomaly"], -1, 1)
        b1, b2 = st.columns(2)
        b1.image(_to_uint8_rgba(ndvi_to_rgba(baseline_ndvi_arr)), caption="3-yr seasonal baseline (reconstructed)", use_container_width=True)
        b2.image(_to_uint8_rgba(ndvi_to_rgba(result["ndvi"])), caption="Current", use_container_width=True)
        anomaly = result["ndvi_anomaly"]
        st.caption(f"Anomaly: {anomaly:+.2f} NDVI vs. baseline "
                   f"({'unusually stressed' if anomaly < -0.08 else 'unusually vigorous' if anomaly > 0.08 else 'within normal range'}).")

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
    st.markdown(result["report_markdown"])

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
        st.warning(f"Couldn't generate the PDF for {label} ({e}). The report above is still complete.")

    if pdf_bytes is not None:
        st.download_button(
            f"⬇️ Download {label} report as PDF", data=pdf_bytes,
            file_name=f"crop_health_report_{label.replace(' ', '_').lower()}.pdf",
            mime="application/pdf", key=f"pdf_{index}",
        )


with st.sidebar:
    st.subheader("📜 Run history")
    history = load_history()
    if not history:
        st.caption("No runs saved yet. Run an analysis to see it here.")
    else:
        for entry in history[:15]:
            with st.expander(f"{entry.get('label', 'Area')} — {entry.get('saved_at', '')[:16]}"):
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
