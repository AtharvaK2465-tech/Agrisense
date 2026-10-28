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
import hashlib
from utils.geo import validate_polygon, get_aoi_metadata
from agents.state import PipelineState


# ---------------------------------------------------------------------------
# Official ESA / Copernicus Sentinel-2 L2A Scene Classification Layer (SCL)
# Reference: Sentinel-2 Products Specification Document (PSD), ESA / Sentinel Hub
# ---------------------------------------------------------------------------
SCL_NO_DATA = 0                      # Missing / nodata / out-of-swath
SCL_SATURATED_OR_DEFECTIVE = 1       # Saturated or defective sensor pixels
SCL_CAST_SHADOW = 2                  # Topographic cast shadows / dark features
SCL_CLOUD_SHADOWS = 3                # Cloud shadows cast onto ground
SCL_VEGETATION = 4                   # Clear green vegetation
SCL_NOT_VEGETATED = 5                # Bare soils, rocks, built-up areas
SCL_WATER = 6                        # Water bodies (valid reflectance, but non-crop)
SCL_UNCLASSIFIED = 7                 # Unclassified surface
SCL_CLOUD_MEDIUM_PROBABILITY = 8     # Medium cloud probability
SCL_CLOUD_HIGH_PROBABILITY = 9       # High cloud probability / thick cloud
SCL_THIN_CIRRUS = 10                 # Thin cirrus clouds
SCL_SNOW_ICE = 11                    # Snow or ice cover

SCL_CLOUD_CLASSES = (SCL_CLOUD_MEDIUM_PROBABILITY, SCL_CLOUD_HIGH_PROBABILITY, SCL_THIN_CIRRUS)
SCL_SHADOW_CLASSES = (SCL_CAST_SHADOW, SCL_CLOUD_SHADOWS)
SCL_DEFECTIVE_CLASSES = (SCL_SATURATED_OR_DEFECTIVE,)
SCL_NODATA_CLASSES = (SCL_NO_DATA,)


def _compute_aoi_mask(
    polygon_geojson: Optional[dict],
    shape_hw: Tuple[int, int],
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Computes a boolean mask of shape (H, W) indicating whether each pixel center
    falls inside the requested AOI polygon. Pixels outside are off-AOI."""
    h, w = shape_hw
    if polygon_geojson is None:
        return np.ones((h, w), dtype=bool)
    try:
        from shapely.geometry import shape
        import shapely
        geom = shape(polygon_geojson)
        if bbox is None:
            minx, miny, maxx, maxy = geom.bounds
        else:
            minx, miny, maxx, maxy = bbox
        if maxx <= minx or maxy <= miny or h == 0 or w == 0:
            return np.ones((h, w), dtype=bool)
        dx = (maxx - minx) / w
        dy = (maxy - miny) / h
        xs = np.linspace(minx + dx / 2.0, maxx - dx / 2.0, w)
        ys = np.linspace(maxy - dy / 2.0, miny + dy / 2.0, h)
        xx, yy = np.meshgrid(xs, ys)
        mask = shapely.contains_xy(geom, xx, yy)
        if not np.any(mask):
            return np.ones((h, w), dtype=bool)
        return mask.astype(bool)
    except Exception:
        return np.ones((h, w), dtype=bool)


def evaluate_pixel_quality(
    raster: np.ndarray,
    polygon_geojson: Optional[dict] = None,
    scl: Optional[np.ndarray] = None,
    data_mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluates pixel-level data quality, distinguishing:
    1. No-data or off-swath pixels (dataMask == 0, SCL == 0, or all bands NaN).
    2. Pixels outside the requested AOI (off-polygon within grid).
    3. Cloud-contaminated pixels (SCL 8, 9, 10).
    4. Cloud-shadow-contaminated pixels (SCL 2, 3).
    5. Pixels with invalid or non-finite band values.
    6. Pixels with valid measurements but potentially unreliable vegetation-index interpretation (SCL 1, 6, 11).

    Returns explicit pixel masks and separate coverage metrics with clearly stated denominators.
    """
    h, w = raster.shape[0], raster.shape[1]
    n_grid = h * w

    # Extract SCL and data_mask from raster if not passed explicitly
    if scl is None and raster.shape[-1] >= 7:
        scl = raster[..., 6]
    if data_mask is None and raster.shape[-1] >= 8:
        data_mask = raster[..., 7]

    # 2. AOI mask (inside vs outside requested AOI)
    aoi_mask = _compute_aoi_mask(polygon_geojson, (h, w), bbox=bbox)
    n_aoi = int(np.sum(aoi_mask))
    if n_aoi == 0:
        aoi_mask = np.ones((h, w), dtype=bool)
        n_aoi = n_grid

    # 1. No-data / off-swath mask
    num_spectral = min(raster.shape[-1], 6)
    all_bands_nan = np.all(~np.isfinite(raster[..., :num_spectral]), axis=-1)
    if data_mask is not None:
        dm_invalid = (data_mask < 0.5)
    else:
        dm_invalid = np.zeros((h, w), dtype=bool)

    if scl is not None:
        scl_nodata = (scl == SCL_NO_DATA)
    else:
        scl_nodata = np.zeros((h, w), dtype=bool)

    nodata_mask = aoi_mask & (dm_invalid | scl_nodata | all_bands_nan)
    n_nodata = int(np.sum(nodata_mask))

    # Source data availability: pixels inside AOI with satellite data acquisition
    source_data_valid = aoi_mask & (~nodata_mask)
    n_data_valid = int(np.sum(source_data_valid))

    # 5. Invalid or non-finite band values in acquired data
    red_idx = 0
    nir_idx = min(3, raster.shape[-1] - 1)
    invalid_bands_mask = source_data_valid & (
        (~np.isfinite(raster[..., red_idx])) | (raster[..., red_idx] < 0.0) |
        (~np.isfinite(raster[..., nir_idx])) | (raster[..., nir_idx] < 0.0)
    )
    n_invalid_bands = int(np.sum(invalid_bands_mask))

    # 3. Cloud contamination
    if scl is not None:
        cloud_mask = source_data_valid & np.isin(scl, SCL_CLOUD_CLASSES)
        n_cloud = int(np.sum(cloud_mask))
        cloud_status = "active"
    else:
        cloud_mask = np.zeros((h, w), dtype=bool)
        n_cloud = None
        cloud_status = "unavailable"

    # 4. Shadow contamination
    if scl is not None:
        shadow_mask = source_data_valid & np.isin(scl, SCL_SHADOW_CLASSES)
        n_shadow = int(np.sum(shadow_mask))
    else:
        shadow_mask = np.zeros((h, w), dtype=bool)
        n_shadow = None

    # 6. Unreliable for vegetation indices (Saturated/defective, Water, Snow)
    if scl is not None:
        defective_mask = source_data_valid & np.isin(scl, SCL_DEFECTIVE_CLASSES)
        water_mask = source_data_valid & (scl == SCL_WATER)
        snow_mask = source_data_valid & (scl == SCL_SNOW_ICE)
        unreliable_mask = defective_mask | water_mask | snow_mask
        n_unreliable = int(np.sum(unreliable_mask))
    else:
        defective_mask = np.zeros((h, w), dtype=bool)
        water_mask = np.zeros((h, w), dtype=bool)
        snow_mask = np.zeros((h, w), dtype=bool)
        unreliable_mask = np.zeros((h, w), dtype=bool)
        n_unreliable = None

    # Usable pixels for vegetation index calculation:
    # Must be inside AOI, valid swath, finite reflectance, and free of cloud/shadow/defective pixels.
    if scl is not None:
        excluded_for_indices = nodata_mask | invalid_bands_mask | cloud_mask | shadow_mask | defective_mask
        usable_mask = aoi_mask & (~excluded_for_indices)
    else:
        usable_mask = aoi_mask & (~nodata_mask) & (~invalid_bands_mask)
    n_usable = int(np.sum(usable_mask))

    # Fractions and explicit denominators
    aoi_coverage_pct = round((n_aoi / n_grid * 100.0), 2) if n_grid > 0 else 0.0
    source_avail_pct = round((n_data_valid / n_aoi * 100.0), 2) if n_aoi > 0 else 0.0
    nodata_pct = round((n_nodata / n_aoi * 100.0), 2) if n_aoi > 0 else 0.0

    if scl is not None:
        cloud_pct = round((n_cloud / n_aoi * 100.0), 2) if n_aoi > 0 else 0.0
        shadow_pct = round((n_shadow / n_aoi * 100.0), 2) if n_aoi > 0 else 0.0
        cloud_shadow_pct = round(((n_cloud + n_shadow) / n_aoi * 100.0), 2) if n_aoi > 0 else 0.0
        cloud_reason = "Sentinel-2 L2A SCL classes 8 (Medium Cloud), 9 (High Cloud), 10 (Thin Cirrus)"
        shadow_reason = "Sentinel-2 L2A SCL classes 2 (Cast Shadow), 3 (Cloud Shadow)"
    else:
        cloud_pct = None
        shadow_pct = None
        cloud_shadow_pct = None
        cloud_reason = "Scene Classification Layer (SCL) not present; pixel-level cloud masking unavailable."
        shadow_reason = "Scene Classification Layer (SCL) not present; shadow masking unavailable."

    usable_pct = round((n_usable / n_aoi * 100.0), 2) if n_aoi > 0 else 0.0
    usable_of_acquired_pct = round((n_usable / n_data_valid * 100.0), 2) if n_data_valid > 0 else 0.0

    return {
        "masks": {
            "aoi_mask": aoi_mask,
            "nodata_mask": nodata_mask,
            "cloud_mask": cloud_mask,
            "shadow_mask": shadow_mask,
            "invalid_bands_mask": invalid_bands_mask,
            "unreliable_mask": unreliable_mask,
            "usable_mask": usable_mask,
        },
        "counts": {
            "grid_pixels": n_grid,
            "aoi_pixels": n_aoi,
            "nodata_pixels": n_nodata,
            "source_data_valid_pixels": n_data_valid,
            "invalid_band_pixels": n_invalid_bands,
            "cloud_pixels": n_cloud,
            "shadow_pixels": n_shadow,
            "usable_pixels": n_usable,
        },
        "metrics": {
            "aoi_pixel_coverage_pct": aoi_coverage_pct,
            "aoi_coverage_denominator": "Total raster grid pixels (H x W)",
            "source_data_availability_pct": source_avail_pct,
            "source_data_availability_denominator": "Total AOI pixels",
            "nodata_fraction_pct": nodata_pct,
            "nodata_denominator": "Total AOI pixels",
            "cloud_fraction_pct": cloud_pct,
            "cloud_denominator": "Total AOI pixels" if cloud_pct is not None else "unavailable (SCL missing)",
            "shadow_fraction_pct": shadow_pct,
            "shadow_denominator": "Total AOI pixels" if shadow_pct is not None else "unavailable (SCL missing)",
            "cloud_shadow_fraction_pct": cloud_shadow_pct,
            "cloud_shadow_denominator": "Total AOI pixels" if cloud_shadow_pct is not None else "unavailable (SCL missing)",
            "usable_pixel_pct": usable_pct,
            "usable_pixel_denominator": "Total AOI pixels",
            "usable_of_acquired_data_pct": usable_of_acquired_pct,
            "usable_of_acquired_denominator": "Acquired valid data pixels within AOI",
            "cloud_masking_status": cloud_status,
            "cloud_masking_reason": cloud_reason,
            "shadow_masking_reason": shadow_reason,
            "masking_method": "ESA Sen2Cor SCL (classes 2,3,8,9,10) + dataMask + AOI Geometry" if scl is not None else "dataMask / band-validity only (SCL unavailable)",
        },
    }


def _mock_fetch_imagery(state: PipelineState) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Generates a deterministic 8-band (Red, Green, Blue, NIR, SWIR, RedEdge, SCL, dataMask)
    synthetic raster seeded by the polygon, clearly labeled as synthetic in metadata."""
    h = hashlib.md5(str(state["polygon_geojson"]).encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.default_rng(seed)

    from utils.districts import detect_ground_context
    ground_info = detect_ground_context(state["polygon_geojson"])

    h_dim, w_dim = 128, 128
    category = ground_info.get("category", "")
    if category == "Water Body":
        # Water: low NIR reflectance, moderate blue/green, negative NDVI (-0.6 to -0.2)
        red = rng.uniform(0.04, 0.08, (h_dim, w_dim)).astype(np.float32)
        green = rng.uniform(0.10, 0.18, (h_dim, w_dim)).astype(np.float32)
        blue = rng.uniform(0.12, 0.22, (h_dim, w_dim)).astype(np.float32)
        nir = rng.uniform(0.01, 0.04, (h_dim, w_dim)).astype(np.float32)
        swir = rng.uniform(0.01, 0.03, (h_dim, w_dim)).astype(np.float32)
        rededge = rng.uniform(0.03, 0.07, (h_dim, w_dim)).astype(np.float32)
        scl = np.full((h_dim, w_dim), SCL_WATER, dtype=np.float32)
    elif ground_info.get("is_non_agri"):
        # Built-up, campus ground (like VIT ground), road asphalt, or bare urban soil
        red = rng.uniform(0.22, 0.32, (h_dim, w_dim)).astype(np.float32)
        green = rng.uniform(0.18, 0.27, (h_dim, w_dim)).astype(np.float32)
        blue = rng.uniform(0.15, 0.24, (h_dim, w_dim)).astype(np.float32)
        nir = rng.uniform(0.20, 0.28, (h_dim, w_dim)).astype(np.float32)
        swir = rng.uniform(0.25, 0.35, (h_dim, w_dim)).astype(np.float32)
        rededge = rng.uniform(0.21, 0.29, (h_dim, w_dim)).astype(np.float32)
        scl = np.full((h_dim, w_dim), SCL_NOT_VEGETATED, dtype=np.float32)
    else:
        # Agricultural field with vegetative canopy
        red = rng.uniform(0.05, 0.25, (h_dim, w_dim)).astype(np.float32)
        green = rng.uniform(0.05, 0.30, (h_dim, w_dim)).astype(np.float32)
        blue = rng.uniform(0.05, 0.20, (h_dim, w_dim)).astype(np.float32)
        nir = rng.uniform(0.20, 0.55, (h_dim, w_dim)).astype(np.float32)
        swir = rng.uniform(0.10, 0.30, (h_dim, w_dim)).astype(np.float32)
        rededge = rng.uniform(0.15, 0.40, (h_dim, w_dim)).astype(np.float32)
        scl = np.full((h_dim, w_dim), SCL_VEGETATION, dtype=np.float32)

        # Seeded stressed patch for downstream agent evaluation
        cy, cx = rng.integers(30, 98, 2)
        yy, xx = np.ogrid[:h_dim, :w_dim]
        patch = (yy - cy) ** 2 + (xx - cx) ** 2 <= 20 ** 2
        nir[patch] *= 0.5
        red[patch] *= 1.3
        swir[patch] *= 1.2
        rededge[patch] *= 0.7

    data_mask = np.ones((h_dim, w_dim), dtype=np.float32)
    raster = np.stack([red, green, blue, nir, swir, rededge, scl, data_mask], axis=-1)
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
        "resolution_m": 10.0,
        "bands_requested": ["B04", "B03", "B02", "B08", "B11", "B05", "SCL", "dataMask"],
        "resampling_note": "Synthetic 10m mock raster grid with simulated SCL and dataMask",
    }
    return raster, cloud_cover, scene_meta


_EVALSCRIPT_8BAND = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B05", "B08", "B11", "SCL", "dataMask"],
    output: { bands: 8, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  // If pixel is outside acquisition swath or nodata, return NaN across spectral bands
  if (sample.dataMask === 0) {
    return [NaN, NaN, NaN, NaN, NaN, NaN, 0, 0];
  }
  // order: red (B04 665nm), green (B03 560nm), blue (B02 490nm), nir (B08 842nm), swir (B11 1610nm), rededge (B05 705nm), SCL, dataMask
  return [sample.B04, sample.B03, sample.B02, sample.B08, sample.B11, sample.B05, sample.SCL, sample.dataMask];
}
"""

_EVALSCRIPT_6BAND = _EVALSCRIPT_8BAND


def _bbox_from_polygon(polygon_geojson: dict):
    from shapely.geometry import shape
    return shape(polygon_geojson).bounds  # (minx, miny, maxx, maxy)


def _sanitize_error_msg(exc: Exception, client_id: str, client_secret: str) -> str:
    """Masks credentials from exception messages to prevent secret leaks in logs."""
    msg = str(exc)
    if client_id:
        msg = msg.replace(client_id, "***")
    if client_secret:
        msg = msg.replace(client_secret, "***")
    return msg


def _real_fetch_imagery(state: PipelineState) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """LIVE MODE — queries Sentinel Hub's Catalog API for the least-cloudy
    Sentinel-2 L2A scene in the requested date range over the drawn polygon,
    then fetches a 6-band clip (RGB, NIR, SWIR, red-edge) via the Process API.

    Spatial Resolution & Resampling:
      - Bands B02, B03, B04, B08 have native 10m resolution.
      - Bands B05 (RedEdge) and B11 (SWIR) have native 20m resolution.
      - Sentinel Hub server-side processing resamples 20m bands to the requested 10m resolution.
      - dataMask is inspected: nodata/out-of-swath pixels return NaN.
    """
    try:
        from sentinelhub import (
            SHConfig, BBox, CRS, SentinelHubRequest, SentinelHubCatalog,
            DataCollection, MimeType, bbox_to_dimensions,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The 'sentinelhub' Python package is required for Live Mode. "
            "Install with 'pip install sentinelhub'."
        ) from exc

    client_id, client_secret = config.get_sentinel_hub_credentials()

    if not (client_id and client_secret):
        raise ValueError(
            "Missing Sentinel Hub credentials (SENTINEL_HUB_CLIENT_ID or SENTINEL_HUB_CLIENT_SECRET). "
            "Please configure credentials in .env or environment variables to enable Live Mode."
        )

    sh_config = SHConfig()
    sh_config.sh_client_id = client_id
    sh_config.sh_client_secret = client_secret
    sh_config.download_timeout_seconds = 30
    sh_config.number_of_retries = 2

    minx, miny, maxx, maxy = _bbox_from_polygon(state["polygon_geojson"])
    bbox = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)

    lon_span_km = (maxx - minx) * 105.0
    lat_span_km = (maxy - miny) * 111.0
    span_km = max(lon_span_km, lat_span_km, 0.5)
    resolution = max((span_km * 1000) / 512, 10.0)  # meters/pixel, floor at native 10m
    size = bbox_to_dimensions(bbox, resolution=resolution)
    size = (min(size[0], 1024), min(size[1], 1024))

    start_date = state["date_start"]
    end_date = state["date_end"]
    time_interval = (start_date.isoformat(), end_date.isoformat())

    try:
        catalog = SentinelHubCatalog(config=sh_config)
        search_results = list(catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=bbox,
            time=time_interval,
            fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover", "properties.grid:code"], "exclude": []},
        ))
    except Exception as exc:
        clean_msg = _sanitize_error_msg(exc, client_id, client_secret)
        if "401" in clean_msg or "403" in clean_msg or "InvalidClient" in clean_msg:
            raise PermissionError(
                "Sentinel Hub authentication failed (HTTP 401/403). "
                "Verify SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET in .env."
            ) from None
        elif "429" in clean_msg or "RateLimit" in clean_msg:
            raise RuntimeError("Sentinel Hub rate limit reached (HTTP 429). Please wait before retrying.") from None
        elif "500" in clean_msg or "502" in clean_msg or "503" in clean_msg:
            raise RuntimeError("Sentinel Hub service temporarily unavailable (HTTP 5xx). Copernicus servers may be busy.") from None
        elif "timeout" in clean_msg.lower() or "timed out" in clean_msg.lower():
            raise TimeoutError("Sentinel Hub request timed out. Check network connectivity.") from None
        raise RuntimeError(f"Sentinel Hub catalog search failed: {clean_msg}") from None

    if not search_results:
        raise RuntimeError(
            f"No qualifying Sentinel-2 scenes found for this AOI between "
            f"{start_date} and {end_date}. Try widening the date range."
        )

    best_scene = min(search_results, key=lambda r: r["properties"].get("eo:cloud_cover", 100))
    cloud_cover = float(best_scene["properties"].get("eo:cloud_cover", 100))
    scene_id = best_scene.get("id") or best_scene.get("properties", {}).get("id", "S2-L2A-LIVE")
    scene_datetime = best_scene.get("properties", {}).get("datetime", end_date.isoformat())
    tile = best_scene.get("properties", {}).get("grid:code", "Sentinel-2 MGRS")

    try:
        request = SentinelHubRequest(
            evalscript=_EVALSCRIPT_8BAND,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order="leastCC",
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=sh_config,
        )
        data_arrays = request.get_data()
    except Exception as exc:
        clean_msg = _sanitize_error_msg(exc, client_id, client_secret)
        if "401" in clean_msg or "403" in clean_msg or "InvalidClient" in clean_msg:
            raise PermissionError(
                "Sentinel Hub Process API authentication failed (HTTP 401/403). "
                "Check SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET in .env."
            ) from None
        elif "429" in clean_msg or "RateLimit" in clean_msg:
            raise RuntimeError("Sentinel Hub rate limit reached (HTTP 429). Please wait before retrying.") from None
        elif "500" in clean_msg or "502" in clean_msg or "503" in clean_msg:
            raise RuntimeError("Sentinel Hub Process API error (HTTP 5xx). Service may be temporarily unavailable.") from None
        elif "timeout" in clean_msg.lower() or "timed out" in clean_msg.lower():
            raise TimeoutError("Sentinel Hub request timed out. Check network connectivity.") from None
        raise RuntimeError(f"Sentinel Hub Process API fetch failed: {clean_msg}") from None

    if not data_arrays or len(data_arrays) == 0:
        raise RuntimeError("Sentinel Hub Process API returned an empty data response.")

    raster = data_arrays[0].astype(np.float32)

    # Validate raster dimensions
    if raster.ndim != 3 or raster.shape[-1] < 6:
        raise ValueError(f"Sentinel Hub returned unexpected raster shape: {raster.shape}, expected (H, W, 8) or (H, W, >=6).")

    # Evaluate pixel quality and cloud/shadow masking
    quality_eval = evaluate_pixel_quality(
        raster,
        polygon_geojson=state.get("polygon_geojson"),
        bbox=(minx, miny, maxx, maxy),
    )
    usable_fraction = quality_eval["metrics"]["usable_pixel_pct"] / 100.0

    if usable_fraction < 0.01:
        raise RuntimeError(
            f"Retrieved Sentinel-2 imagery contains insufficient valid data ({usable_fraction:.1%} usable pixels). "
            "The AOI may be fully cloud-obscured, shadow-contaminated, or fall outside the orbit swath boundary."
        )

    bands_requested = ["B04", "B03", "B02", "B08", "B11", "B05", "SCL", "dataMask"] if raster.shape[-1] >= 8 else ["B04", "B03", "B02", "B08", "B11", "B05"]
    scene_meta = {
        "scene_id": scene_id,
        "datetime": scene_datetime,
        "tile": tile,
        "provider": "Copernicus Sentinel-2 L2A (Sentinel Hub Process API)",
        "cloud_cover": round(cloud_cover, 1),
        "is_synthetic": False,
        "resolution_m": round(resolution, 1),
        "bands_requested": bands_requested,
        "resampling_note": "Sentinel-2 20m bands (B05, B11, SCL) resampled to 10m resolution via Sentinel Hub server-side interpolation",
        "valid_pixel_ratio": round(usable_fraction, 3),
        "quality_metrics": quality_eval["metrics"],
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


DEFAULT_S2_BAND_MAPPING: Dict[str, int] = {
    "red": 0,       # Sentinel-2 B04 (Red, 665nm, 10m native resolution)
    "green": 1,     # Sentinel-2 B03 (Green, 560nm, 10m native resolution)
    "blue": 2,      # Sentinel-2 B02 (Blue, 490nm, 10m native resolution)
    "nir": 3,       # Sentinel-2 B08 (NIR, 842nm, 10m native resolution)
    "swir": 4,      # Sentinel-2 B11 (SWIR-1, 1610nm, 20m resampled to 10m)
    "rededge": 5,   # Sentinel-2 B05 (RedEdge-1, 705nm, 20m resampled to 10m)
}


def _compute_indices(
    raster: np.ndarray,
    is_synthetic: bool = False,
    band_mapping: Optional[Dict[str, int]] = None,
    polygon_geojson: Optional[dict] = None,
    scl: Optional[np.ndarray] = None,
    data_mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Calculates spectral indices (NDVI, NDMI, NDRE) and surface classification fractions
    strictly masked by pixel quality (excluding clouds, cloud shadows, nodata, and non-finite pixels).
    
    Exact Sentinel-2 Band Mapping:
      - NDVI: NIR (B08, 842nm) and Red (B04, 665nm) -> (NIR - Red) / (NIR + Red)
      - NDMI: NIR (B08, 842nm) and SWIR (B11, 1610nm) -> (NIR - SWIR) / (NIR + SWIR)
      - NDRE: NIR (B08, 842nm) and Red-Edge (B05, 705nm) -> (NIR - RedEdge) / (NIR + RedEdge)
    
    Safeguards:
      - If required bands are missing or out of bounds, index is marked 'unavailable' with explicit reason.
      - Never calculates an index from an incorrect band just because an array position exists.
      - Masked pixels (cloud, shadow, nodata) are converted to NaN so they cannot silently contaminate
        vegetation index summaries or crop classifications.
      - If zero usable clear pixels exist, status is marked 'insufficient_usable_pixels'.
    """
    if raster.ndim != 3 or raster.shape[0] == 0 or raster.shape[1] == 0:
        raise ValueError(f"Raster requires a 3D array (H, W, Bands) with positive dimensions, got shape {raster.shape}.")

    mapping = band_mapping if band_mapping is not None else DEFAULT_S2_BAND_MAPPING
    num_bands = raster.shape[-1]
    indices_quality: Dict[str, Any] = {}

    # Evaluate comprehensive pixel quality across all 6 categories
    quality_eval = evaluate_pixel_quality(
        raster,
        polygon_geojson=polygon_geojson,
        scl=scl,
        data_mask=data_mask,
        bbox=bbox,
    )
    usable_mask = quality_eval["masks"]["usable_mask"]
    n_usable = quality_eval["counts"]["usable_pixels"]
    n_aoi = quality_eval["counts"]["aoi_pixels"]

    # 1. NDVI: NIR and Red
    if "nir" in mapping and "red" in mapping and mapping["nir"] < num_bands and mapping["red"] < num_bands:
        red = raster[..., mapping["red"]].copy()
        nir = raster[..., mapping["nir"]].copy()
        ndvi = _safe_normalized_difference(nir, red)
        # Mask out unusable pixels (clouds, shadows, nodata, non-finite)
        ndvi[~usable_mask] = np.nan

        if n_usable > 0:
            indices_quality["ndvi"] = {
                "status": "calculated",
                "bands": "B08 (NIR 842nm), B04 (Red 665nm)",
                "formula": "(NIR - Red) / (NIR + Red)",
                "is_synthetic": is_synthetic,
            }
        else:
            indices_quality["ndvi"] = {
                "status": "insufficient_usable_pixels",
                "reason": "Zero usable clear-sky pixels within AOI (fully cloud-obscured, shadowed, or nodata).",
                "is_synthetic": is_synthetic,
            }
    else:
        ndvi = None
        missing = []
        if "nir" not in mapping or mapping.get("nir", 999) >= num_bands:
            missing.append("NIR (B08)")
        if "red" not in mapping or mapping.get("red", 999) >= num_bands:
            missing.append("Red (B04)")
        indices_quality["ndvi"] = {
            "status": "unavailable",
            "reason": f"Required band(s) missing or out of bounds: {', '.join(missing)}.",
            "is_synthetic": is_synthetic,
        }

    # 2. NDMI: NIR and SWIR (Sentinel-2 B08 and B11)
    if "nir" in mapping and "swir" in mapping and mapping["nir"] < num_bands and mapping["swir"] < num_bands:
        swir = raster[..., mapping["swir"]].copy()
        nir = raster[..., mapping["nir"]].copy()
        ndmi = _safe_normalized_difference(nir, swir)
        ndmi[~usable_mask] = np.nan
        if n_usable > 0:
            indices_quality["ndmi"] = {
                "status": "calculated",
                "bands": "B08 (NIR 842nm), B11 (SWIR 1610nm)",
                "formula": "(NIR - SWIR) / (NIR + SWIR)",
                "is_synthetic": is_synthetic,
            }
        else:
            indices_quality["ndmi"] = {
                "status": "insufficient_usable_pixels",
                "reason": "Zero usable clear-sky pixels within AOI.",
                "is_synthetic": is_synthetic,
            }
    else:
        ndmi = None
        indices_quality["ndmi"] = {
            "status": "unavailable",
            "reason": "SWIR band (B11 1610nm) not present in band mapping or raster dimensions.",
            "is_synthetic": is_synthetic,
        }

    # 3. NDRE: NIR and Red-Edge (Sentinel-2 B08 and B05)
    if "nir" in mapping and "rededge" in mapping and mapping["nir"] < num_bands and mapping["rededge"] < num_bands:
        rededge = raster[..., mapping["rededge"]].copy()
        nir = raster[..., mapping["nir"]].copy()
        ndre = _safe_normalized_difference(nir, rededge)
        ndre[~usable_mask] = np.nan
        if n_usable > 0:
            indices_quality["ndre"] = {
                "status": "calculated",
                "bands": "B08 (NIR 842nm), B05 (RedEdge 705nm)",
                "formula": "(NIR - RedEdge) / (NIR + RedEdge)",
                "is_synthetic": is_synthetic,
            }
        else:
            indices_quality["ndre"] = {
                "status": "insufficient_usable_pixels",
                "reason": "Zero usable clear-sky pixels within AOI.",
                "is_synthetic": is_synthetic,
            }
    else:
        ndre = None
        indices_quality["ndre"] = {
            "status": "unavailable",
            "reason": "Red-Edge band (B05 705nm) not present in band mapping or raster dimensions.",
            "is_synthetic": is_synthetic,
        }

    indices_quality["cloud_masking"] = {
        "status": quality_eval["metrics"]["cloud_masking_status"],
        "reason": quality_eval["metrics"]["cloud_masking_reason"],
        "method": quality_eval["metrics"]["masking_method"],
    }

    # Surface classification fractions across AOI pixels
    if ndvi is not None and n_usable > 0:
        usable_ndvi = ndvi[usable_mask]
        veg_count = int(np.sum(usable_ndvi >= 0.25))
        bare_count = int(np.sum((usable_ndvi >= 0.0) & (usable_ndvi < 0.25)))
        water_count = int(np.sum(usable_ndvi < 0.0))
        unclassified_count = n_aoi - n_usable

        surface_fractions = {
            "vegetation": round(float(veg_count / n_aoi), 3) if n_aoi > 0 else 0.0,
            "bare_surface": round(float(bare_count / n_aoi), 3) if n_aoi > 0 else 0.0,
            "water_surface": round(float(water_count / n_aoi), 3) if n_aoi > 0 else 0.0,
            "unclassified": round(float(unclassified_count / n_aoi), 3) if n_aoi > 0 else 0.0,
        }
    else:
        surface_fractions = {
            "vegetation": 0.0,
            "bare_surface": 0.0,
            "water_surface": 0.0,
            "unclassified": 1.0,
        }

    valid_pct = quality_eval["metrics"]["usable_pixel_pct"] if ndvi is not None else 0.0

    return {
        "ndvi": ndvi,
        "ndmi": ndmi,
        "ndre": ndre,
        "indices_quality": indices_quality,
        "surface_fractions": surface_fractions,
        "valid_pixels_pct": valid_pct,
        "quality_metrics": quality_eval["metrics"],
        "pixel_counts": quality_eval["counts"],
        "usable_mask": usable_mask,
    }


def geo_agent(state: PipelineState) -> PipelineState:
    is_valid, message = validate_polygon(state["polygon_geojson"])
    if not is_valid:
        return {**state, "error": message}

    aoi_meta = get_aoi_metadata(state["polygon_geojson"])
    attempts = state.get("query_attempts", 0) + 1
    geo_fallback_note = None

    is_mock = config.is_sentinel_hub_mock() if hasattr(config, "is_sentinel_hub_mock") else getattr(config, "SENTINEL_HUB_MOCK", True)
    if is_mock:
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

    indices = _compute_indices(
        raster,
        is_synthetic=is_synthetic,
        polygon_geojson=state.get("polygon_geojson"),
    )

    quality_m = indices.get("quality_metrics", {})
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
        "resolution_m": scene_meta.get("resolution_m"),
        "resampling_note": scene_meta.get("resampling_note"),
        "quality_metrics": quality_m,
        "pixel_quality_metrics": quality_m,
        "masking_method": quality_m.get("masking_method", "ESA Sen2Cor SCL + dataMask"),
        "scl_classes_excluded": {
            "cloud": [8, 9, 10],
            "shadow": [2, 3],
            "nodata_defective": [0, 1],
        },
        "masking_disclaimer": (
            "Pixel-level cloud masking is derived from the Sentinel-2 L2A Scene Classification Layer (SCL) "
            "at 20m native resolution resampled to 10m. SCL is an automated algorithmic classification "
            "and does not provide 100% cloud detection accuracy; residual thin cirrus or cloud edges may occasionally remain undetected."
        ),
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
        "quality_metrics": quality_m,
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
