# Multi-Agent Crop Health Monitoring — Maharashtra

Satellite-imagery-based crop disease/stress detection for a user-selected
area of Maharashtra, built as a LangGraph multi-agent pipeline with a
Streamlit + map-based UI for area selection.

## How it works

1. You draw one or more polygons/rectangles on the map (Leaflet, via
   `streamlit-folium`) — each is analyzed and reported separately, with a
   side-by-side comparison table when there's more than one.
2. **Geo/Area Agent** fetches Sentinel-2 imagery clipped to each polygon and
   computes NDVI / NDMI / NDRE.
3. **Input/Validation Agent** checks cloud cover. If it's too high, it loops
   back to the Geo Agent with a shifted date range (a real conditional edge
   in the graph — not just a linear pipeline).
4. **Crop-Type Agent** samples NDVI at 4 phenology stages across the growing
   season and matches the curve shape against reference signatures for
   cotton, soybean, sugarcane, jowar and tur — a training-data-free
   technique, not a black box. The result is cross-checked against the
   district's Kharif/Rabi crop calendar and re-ranked if implausible. It
   also computes a per-pixel confidence map and a 3-year seasonal NDVI
   baseline/anomaly.
5. **Vision Agent** classifies the area: healthy / water stress / nutrient
   deficiency / disease.
6. **Reasoning Agent** cross-checks that classification against recent
   rainfall (Open-Meteo) to distinguish, e.g., drought stress from disease.
7. **Report Agent** turns the structured diagnosis — plus crop type,
   district/season context, baseline anomaly, and any matching demo
   advisory records — into a readable report (via Gemini, or a template in
   mock mode), downloadable as a PDF.

### UI highlights
- Color-coded crop-type overlay on the Leaflet map (not a flat NDVI
  grayscale), with a legend, an NDVI red→yellow→green colormap layer, and a
  toggleable per-pixel crop-match confidence heatmap.
- NDVI time-series chart per area vs. its 3-year seasonal baseline, plus a
  before/after baseline-vs-current image comparison.
- A "How this was calculated" methodology panel and a validation panel
  showing matching known-issue records (demo reference data — see
  `utils/advisories.py`).
- Run history persisted to `data/history.json` and browsable from the
  sidebar across refreshes.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in real API keys (optional, see below)
streamlit run app.py
```

## Mock mode vs. live mode

**You don't need any API keys to run this.** If `SENTINEL_HUB_CLIENT_ID`/
`SENTINEL_HUB_CLIENT_SECRET` or `GEMINI_API_KEY` are missing from `.env`,
the app automatically runs those pieces in mock mode:

- Sentinel Hub → synthetic but realistic 4-band rasters with a seeded
  "stressed patch," so the vision/reasoning agents always have something
  meaningful to detect.
- Gemini → a clean template-based report instead of an LLM-generated one.

This means you can build, demo, and test the *entire* pipeline — including
the retry loop and the map UI — before you have real API access. Every
place that needs a real key is marked `# TODO: insert API key` in
`agents/geo_agent.py` and `agents/report_agent.py`, with the real
implementation sketched in a docstring right above the `raise
NotImplementedError`.

## Getting real API keys

- **Sentinel Hub**: free trial / student tier at https://www.sentinel-hub.com
- **Gemini**: free tier key at https://aistudio.google.com/apikey

## Project structure

```
agents/
  state.py           # shared state schema passed between all nodes
  geo_agent.py        # fetch + clip imagery, compute indices
  input_agent.py       # cloud-cover validation + retry decision
  crop_agent.py          # phenology-based crop-type classification + district/baseline context
  vision_agent.py          # stress/disease classification
  reasoning_agent.py         # fuses vision output with weather data
  report_agent.py              # generates the final report
  graph.py                       # wires all agents into a LangGraph StateGraph
utils/
  config.py           # env vars, mock-mode flags, Maharashtra bbox
  geo.py               # polygon validation (bounds, area cap)
  weather.py            # Open-Meteo rainfall lookup (no key needed)
  crops.py               # crop phenology reference signatures + classifier
  districts.py             # district lookup + Kharif/Rabi crop calendar
  advisories.py              # demo ground-truth advisory records + matcher
  colormap.py                  # NDVI/confidence array -> RGBA for map overlays
  history.py                     # JSON-file run history (sidebar, persists across refresh)
  pdfexport.py                     # report_markdown -> downloadable PDF
tests/
  test_geo.py          # polygon validation tests
  test_vision.py         # index computation + classification heuristic tests
  test_crops.py            # phenology classifier tests
  test_districts.py          # district lookup + crop calendar tests
app.py                        # Streamlit UI (map + controls + results, multi-area tabs)
```

## Running tests

```bash
pytest tests/ -v
```

## Next steps / extensions

- Swap the NDVI-threshold heuristic in `vision_agent.py` for a trained
  CNN once you have labeled imagery (hook is at `_model_predict`).
- Swap the phenology curve-matching in `crop_agent.py` for a trained
  Random Forest / small CNN over labeled polygons once you have them —
  the district crop-calendar cross-check and confidence-map logic don't
  need to change.
- Replace `_mock_phenology_series` / `_mock_baseline_ndvi` in
  `crop_agent.py` with real multi-date and multi-year Sentinel Hub pulls.
- Replace the placeholder district bounding boxes in `utils/districts.py`
  with real district polygon boundaries, and the demo entries in
  `utils/advisories.py` with a real advisory source (e.g. the state
  agriculture department or data.gov.in).
- Wire real Sentinel Hub / Gemini calls in the two `NotImplementedError`
  stubs once you have API keys.
- Add the optional Alert/Monitoring Agent: re-run the pipeline on a saved
  polygon on a schedule, notify only when `risk_level` crosses a threshold.
- Use real Sentinel-2 band-edge (B05/B8A/B11) values for NDMI/NDRE once
  live imagery is available — they're currently approximated from
  RGB+NIR only, which is a known simplification worth noting in your
  report/viva.
