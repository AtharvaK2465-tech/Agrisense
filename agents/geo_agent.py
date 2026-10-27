"""
Geo/Area Agent
--------------
Takes the polygon the user drew on the map + a date range, validates the AOI,
fetches Sentinel-2 imagery (real Copernicus data or explicit synthetic rasters),
and calculates spectral indices (NDVI, NDMI, NDRE) and surface fractions with
full data-source provenance and mathematical safeguards.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
from datetime import timedelta, date

import utils.config as config
from utils.geo import validate_polygon, get_aoi_metadata
from agents.state import PipelineState


def _mock_fetch_imagery(state: PipelineState) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Generates a deterministic 6-band (Red, Green, Blue, NIR, SWIR, RedEdge)
    synthetic raster seeded by the polygon, clearly labeled as synthetic in metadata."""
    seed = abs(hash(str(state["polygon_geojson"]))) % (2**32)
    rng = np.random.default_rng(seed)

    from utils.districts import detect_ground_context
    ground_info = detect_ground_context(state["polygon_geojson"])

    h, w = 128, 128
    category = ground_info.get("category", "")
    if category == "Water Body":
        # Water: low NIR reflectance, moderate blue/green, negative NDVI (-0.6 to -0.2)
        red = rng.uniform(0.04, 0.08, (h, w)).astype(np.float32)
        green = rng.uniform(0.10, 0.18, (h, w)).astype(np.float32)
        blue = rng.uniform(0.12, 0.22, (h, w)).astype(np.float32)
        nir = rng.uniform(0.01, 0.04, (h, w)).astype(np.float32)
        swir = rng.uniform(0.01, 0.03, (h, w)).astype(np.float32)
        rededge = rng.uniform(0.03, 0.07, (h, w)).astype(np.float32)
    elif ground_info.get("is_non_agri"):
        # Built-up, campus ground (like VIT ground), road asphalt, or bare urban soil
        red = rng.uniform(0.22, 0.32, (h, w)).astype(np.float32)
        green = rng.uniform(0.18, 0.27, (h, w)).astype(np.float32)
        blue = rng.uniform(0.15, 0.24, (h, w)).astype(np.float32)
        nir = rng.uniform(0.20, 0.28, (h, w)).astype(np.float32)
        swir = rng.uniform(0.25, 0.35, (h, w)).astype(np.float32)
        rededge = rng.uniform(0.21, 0.29, (h, w)).astype(np.float32)
    else:
        # Agricultural field with vegetative canopy
        red = rng.uniform(0.05, 0.25, (h, w)).astype(np.float32)
        green = rng.uniform(0.05, 0.30, (h, w)).astype(np.float32)
        blue = rng.uniform(0.05, 0.20, (h, w)).astype(np.float32)
        nir = rng.uniform(0.20, 0.55, (h, w)).astype(np.float32)
        swir = rng.uniform(0.10, 0.30, (h, w)).astype(np.float32)
        rededge = rng.uniform(0.15, 0.40, (h, w)).astype(np.float32)

        # Seeded stressed patch for downstream agent evaluation
        cy, cx = rng.integers(30, 98, 2)
        yy, xx = np.ogrid[:h, :w]
        patch = (yy - cy) ** 2 + (xx - cx) ** 2 <= 20 ** 2
        nir[patch] *= 0.5
        red[patch] *= 1.3
        swir[patch] *= 1.2
        rededge[patch] *= 0.7

    raster = np.stack([red, green, blue, nir, swir, rededge], axis=-1)
    cloud_cover = float(rng.uniform(2.0, 28.0))

    end_date = state.get("date_end")
    end_date_str = end_date.isoformat() if isinstance(end_date, date) else str(end_date)
    scene_meta = {
        "scene_id": f"SYNTH-S2L2A-{seed:08X}",
        "datetime": f"{end_date_str}T10:30:00Z",
        "tile": "MOCK-43QEA",
        "provider": "Synthetic Raster Generator (Mock Mode)",
        "cloud_cover": round(cloud_cover, 1),
        "is_synthetic": True,
    }
    return raster, cloud_cover, scene_meta


_EVALSCRIPT_6BAND = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B05", "B08", "B11"],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  // order: red, green, blue, nir, swir(B11), rededge(B05)
  return [sample.B04, sample.B03, sample.B02, sample.B08, sample.B11, sample.B05];
}
"""


def _bbox_from_polygon(polygon_geojson: dict):
    from shapely.geometry import shape
    return shape(polygon_geojson).bounds  # (minx, miny, maxx, maxy)


def _real_fetch_imagery(state: PipelineState) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """LIVE MODE — queries Sentinel Hub's Catalog API for the least-cloudy
    Sentinel-2 L2A scene in the requested date range over the drawn polygon,
    then fetches a 6-band clip (RGB, NIR, SWIR, red-edge)."""
    from sentinelhub import (
        SHConfig, BBox, CRS, SentinelHubRequest, SentinelHubCatalog,
        DataCollection, MimeType, bbox_to_dimensions,
    )
    from utils.config import SENTINEL_HUB_CLIENT_ID, SENTINEL_HUB_CLIENT_SECRET

    if not (SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET):
        raise ValueError("Missing SENTINEL_HUB_CLIENT_ID or SENTINEL_HUB_CLIENT_SECRET.")

    config = SHConfig()
    config.sh_client_id = SENTINEL_HUB_CLIENT_ID
    config.sh_client_secret = SENTINEL_HUB_CLIENT_SECRET

    minx, miny, maxx, maxy = _bbox_from_polygon(state["polygon_geojson"])
    bbox = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)

    lon_span_km = (maxx - minx) * 105.0
    lat_span_km = (maxy - miny) * 111.0
    span_km = max(lon_span_km, lat_span_km, 0.5)
    resolution = max((span_km * 1000) / 512, 10)  # meters/pixel, floor at native 10m
    size = bbox_to_dimensions(bbox, resolution=resolution)
    size = (min(size[0], 1024), min(size[1], 1024))

    time_interval = (state["date_start"].isoformat(), state["date_end"].isoformat())

    catalog = SentinelHubCatalog(config=config)
    search_results = list(catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=time_interval,
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover", "properties.grid:code"], "exclude": []},
    ))
    if not search_results:
        raise RuntimeError(
            f"No qualifying Sentinel-2 scenes found for this AOI between "
            f"{state['date_start']} and {state['date_end']}. Try widening the date range."
        )
    best_scene = min(search_results, key=lambda r: r["properties"].get("eo:cloud_cover", 100))
    cloud_cover = float(best_scene["properties"].get("eo:cloud_cover", 100))

    request = SentinelHubRequest(
        evalscript=_EVALSCRIPT_6BAND,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
            mosaicking_order="leastCC",
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    raster = request.get_data()[0].astype(np.float32)
    scene_meta = {
        "scene_id": best_scene.get("id") or best_scene.get("properties", {}).get("id", "S2-L2A-LIVE"),
        "datetime": best_scene.get("properties", {}).get("datetime", state["date_end"].isoformat()),
        "tile": best_scene.get("properties", {}).get("grid:code", "Sentinel-2 MGRS"),
        "provider": "Copernicus Sentinel-2 L2A (Sentinel Hub)",
        "cloud_cover": round(cloud_cover, 1),
        "is_synthetic": False,
    }
    return raster, cloud_cover, scene_meta


def _safe_normalized_difference(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """Computes (band_a - band_b) / (band_a + band_b) safely handling zero denominators,
    non-finite values, and clipping to [-1.0, 1.0]."""
    denom = band_a + band_b
    mask_invalid = (np.abs(denom) < 1e-7) | (~np.isfinite(denom))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mask_invalid, 0.0, (band_a - band_b) / np.where(mask_invalid, 1.0, denom))
    return np.clip(np.nan_to_num(ratio, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)


def _compute_indices(raster: np.ndarray, is_synthetic: bool = False) -> Dict[str, Any]:
    """
    Calculates spectral indices (NDVI, NDMI, NDRE) and surface classification fractions.
    Bands: 0: Red, 1: Green, 2: Blue, 3: NIR, 4: SWIR (B11), 5: RedEdge (B05).
    Safeguard: If required bands are missing, NDMI or NDRE are marked unavailable with an explicit reason.
    """
    if raster.ndim != 3 or raster.shape[-1] < 4:
        raise ValueError(f"Raster requires at least 4 bands (Red, Green, Blue, NIR), got shape {raster.shape}.")

    red = raster[..., 0]
    nir = raster[..., 3]

    # NDVI: (NIR - Red) / (NIR + Red)
    ndvi = _safe_normalized_difference(nir, red)
    indices_quality: Dict[str, Any] = {
        "ndvi": {
            "status": "calculated",
            "bands": "B08 (NIR 842nm), B04 (Red 665nm)",
            "formula": "(NIR - Red) / (NIR + Red)",
            "is_synthetic": is_synthetic,
        }
    }

    # NDMI: (NIR - SWIR) / (NIR + SWIR) -> Sentinel-2 B08 and B11
    if raster.shape[-1] >= 5:
        swir = raster[..., 4]
        ndmi = _safe_normalized_difference(nir, swir)
        indices_quality["ndmi"] = {
            "status": "calculated",
            "bands": "B08 (NIR 842nm), B11 (SWIR 1610nm)",
            "formula": "(NIR - SWIR) / (NIR + SWIR)",
            "is_synthetic": is_synthetic,
        }
    else:
        ndmi = None
        indices_quality["ndmi"] = {
            "status": "unavailable",
            "reason": "SWIR band (B11) not present in 4-band raster.",
            "is_synthetic": False,
        }

    # NDRE: (NIR - RedEdge) / (NIR + RedEdge) -> Sentinel-2 B08 and B05
    if raster.shape[-1] >= 6:
        rededge = raster[..., 5]
        ndre = _safe_normalized_difference(nir, rededge)
        indices_quality["ndre"] = {
            "status": "calculated",
            "bands": "B08 (NIR 842nm), B05 (RedEdge 705nm)",
            "formula": "(NIR - RedEdge) / (NIR + RedEdge)",
            "is_synthetic": is_synthetic,
        }
    else:
        ndre = None
        indices_quality["ndre"] = {
            "status": "unavailable",
            "reason": "Red-Edge band (B05) not present in raster.",
            "is_synthetic": False,
        }

    # Surface classification fractions across valid pixels
    valid_pixels = np.isfinite(ndvi)
    valid_count = int(np.sum(valid_pixels))
    total_count = int(ndvi.size)
    valid_pct = (valid_count / total_count * 100.0) if total_count > 0 else 0.0

    veg_mask = (ndvi >= 0.25) & valid_pixels
    bare_mask = (ndvi >= 0.0) & (ndvi < 0.25) & valid_pixels
    water_mask = (ndvi < 0.0) & valid_pixels

    surface_fractions = {
        "vegetation": round(float(np.mean(veg_mask)), 3),
        "bare_surface": round(float(np.mean(bare_mask)), 3),
        "water_surface": round(float(np.mean(water_mask)), 3),
        "unclassified": round(float(1.0 - (valid_count / total_count)), 3) if total_count > 0 else 0.0,
    }

    return {
        "ndvi": ndvi,
        "ndmi": ndmi,
        "ndre": ndre,
        "indices_quality": indices_quality,
        "surface_fractions": surface_fractions,
        "valid_pixels_pct": round(valid_pct, 2),
    }


def geo_agent(state: PipelineState) -> PipelineState:
    is_valid, message = validate_polygon(state["polygon_geojson"])
    if not is_valid:
        return {**state, "error": message}

    aoi_meta = get_aoi_metadata(state["polygon_geojson"])
    attempts = state.get("query_attempts", 0) + 1
    geo_fallback_note = None

    if config.SENTINEL_HUB_MOCK:
        raster, cloud_cover, scene_meta = _mock_fetch_imagery(state)
        execution_mode = "mock"
        is_synthetic = True
    else:
        try:
            raster, cloud_cover, scene_meta = _real_fetch_imagery(state)
            execution_mode = "live"
            is_synthetic = False
        except Exception as e:
            if not config.ALLOW_LIVE_FALLBACK_TO_MOCK:
                return {
                    **state,
                    "error": (
                        f"Live Sentinel Hub imagery retrieval failed: {e}. "
                        "Live Mode requires valid Sentinel Hub API credentials and qualifying satellite scenes."
                    ),
                }
            raster, cloud_cover, scene_meta = _mock_fetch_imagery(state)
            execution_mode = "live_fallback_to_mock"
            is_synthetic = True
            geo_fallback_note = (
                f"NOTICE: Live Sentinel Hub retrieval failed ({e}). Reverted to synthetic mock raster "
                "for pipeline demonstration. Results are synthetic and must NOT be used as live satellite observations."
            )

    indices = _compute_indices(raster, is_synthetic=is_synthetic)

    provenance = {
        "execution_mode": execution_mode,
        "imagery_source": "synthetic" if is_synthetic else "sentinel-2-l2a",
        "is_synthetic_imagery": is_synthetic,
        "scene_id": scene_meta.get("scene_id"),
        "acquisition_date": scene_meta.get("datetime"),
        "mgrs_tile": scene_meta.get("tile"),
        "provider": scene_meta.get("provider"),
        "cloud_cover_pct": cloud_cover,
        "cloud_threshold_pct": config.MAX_CLOUD_COVER_PCT,
        "aoi": aoi_meta,
        "indices_quality": indices["indices_quality"],
        "surface_fractions": indices["surface_fractions"],
        "valid_pixels_pct": indices["valid_pixels_pct"],
    }

    result: PipelineState = {
        **state,
        "raster": raster,
        "ndvi": indices["ndvi"],
        "ndmi": indices["ndmi"],
        "ndre": indices["ndre"],
        "indices_quality": indices["indices_quality"],
        "surface_fractions": indices["surface_fractions"],
        "valid_pixels_pct": indices["valid_pixels_pct"],
        "cloud_cover_pct": cloud_cover,
        "query_attempts": attempts,
        "provenance": provenance,
    }
    if geo_fallback_note:
        result["geo_fallback_note"] = geo_fallback_note

    return result


def widen_date_range(state: PipelineState) -> PipelineState:
    """Called on the retry edge when cloud cover is too high — shifts the
    search window back to look for a clearer scene."""
    new_start = state["date_start"] - timedelta(days=config.DATE_RANGE_DAYS)
    return {**state, "date_start": new_start}
