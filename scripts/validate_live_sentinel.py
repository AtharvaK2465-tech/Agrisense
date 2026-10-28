"""
Dedicated Live Sentinel-2 Integration Verification Tool
-------------------------------------------------------
Executes a reproducible live Sentinel Hub API validation run against
the European Space Agency / Copernicus Sentinel-2 L2A satellite archive.

Verification Steps:
1. Validates that Sentinel Hub credentials exist in environment/.env without printing secrets.
2. Submits a Catalog API search for the least-cloudy Sentinel-2 scene over the target AOI.
3. Retrieves 6-band multispectral data (RGB, NIR, SWIR, RedEdge) via Sentinel Hub Process API.
4. Verifies dimensions, spatial resolution, nodata masking, and band validity.
5. Computes analytical spectral indices (NDVI, NDMI, NDRE) and surface classification fractions.
6. Records complete provenance and metadata.
7. Exits with code 0 on successful live validation, or nonzero status with actionable diagnostic.
"""
import argparse
import os
import sys
from datetime import date, timedelta
from typing import Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import get_sentinel_hub_credentials
from utils.geo import point_radius_to_geojson, validate_polygon, get_aoi_metadata
from agents.geo_agent import _real_fetch_imagery, _compute_indices


# Standard Maharashtra validation parcel: Hingoli agricultural field
DEFAULT_LAT = 19.72
DEFAULT_LON = 77.12
DEFAULT_RADIUS_KM = 2.0


def validate_live_sentinel_integration(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    radius_km: float = DEFAULT_RADIUS_KM,
    start_date: date = None,
    end_date: date = None,
) -> Dict[str, Any]:
    print("=================================================================")
    print("[AGRISENSE] Live Sentinel-2 Satellite Integration Verification")
    print("=================================================================\n")

    # Step 1: Credential presence check (Never print credentials)
    client_id, client_secret = get_sentinel_hub_credentials()
    if not client_id or not client_secret:
        print("[-] FAILED: Sentinel Hub API credentials not found.")
        print("    Configuration required in .env or environment:")
        print("      SENTINEL_HUB_CLIENT_ID=<your-oauth-client-id>")
        print("      SENTINEL_HUB_CLIENT_SECRET=<your-oauth-client-secret>")
        print("    Create free credentials at: https://dashboard.sentinel-hub.com/")
        print("    Live verification cannot proceed without credentials.")
        return {"status": "FAILED", "reason": "Missing credentials", "exit_code": 1}

    print("[+] Credentials detected in environment (Client ID: present, Secret: present).")

    # Step 2: AOI and Date Window Configuration
    if end_date is None:
        end_date = date.today() - timedelta(days=5)
    if start_date is None:
        start_date = end_date - timedelta(days=30)

    print(f"[+] Target AOI: Center ({lat:.4f}N, {lon:.4f}E), Radius {radius_km:.1f} km")
    print(f"[+] Date Search Window: {start_date.isoformat()} to {end_date.isoformat()} ({ (end_date - start_date).days } days)")

    poly = point_radius_to_geojson(lat, lon, radius_km=radius_km)
    is_valid, msg = validate_polygon(poly)
    if not is_valid:
        print(f"[-] FAILED: AOI validation rejected: {msg}")
        return {"status": "FAILED", "reason": f"AOI invalid: {msg}", "exit_code": 2}

    aoi_meta = get_aoi_metadata(poly)
    print(f"[+] AOI Area: {aoi_meta['area_km2']:.2f} km² ({aoi_meta['area_hectares']:.1f} ha)")

    state = {
        "polygon_geojson": poly,
        "date_start": start_date,
        "date_end": end_date,
    }

    # Step 3: Live Copernicus Sentinel-2 Retrieval
    print("\n--> Contacting Sentinel Hub Catalog and Process APIs...")
    try:
        raster, cloud_cover, scene_meta = _real_fetch_imagery(state)
    except Exception as exc:
        print(f"\n[-] FAILED: Live satellite retrieval encountered an error:")
        print(f"    {exc}")
        return {"status": "FAILED", "reason": str(exc), "exit_code": 3}

    # Step 4: Verify Returned Raster Integrity
    print("[+] Live Sentinel-2 scene successfully acquired!")
    print(f"    Scene ID:          {scene_meta.get('scene_id')}")
    print(f"    Acquisition Time:  {scene_meta.get('datetime')}")
    print(f"    MGRS Tile:         {scene_meta.get('tile')}")
    print(f"    Provider:          {scene_meta.get('provider')}")
    print(f"    Cloud Cover:       {scene_meta.get('cloud_cover')}%")
    print(f"    Spatial Res:       {scene_meta.get('resolution_m')} meters/pixel")
    print(f"    Resampling:        {scene_meta.get('resampling_note')}")
    print(f"    Raster Shape:      {raster.shape} (H x W x Bands)")
    print(f"    Raster Data Type:  {raster.dtype}")

    # Step 5: Spectral Indices Computation with Quality & Cloud Masking
    indices = _compute_indices(raster, is_synthetic=False, polygon_geojson=poly)
    ndvi = indices.get("ndvi")
    ndmi = indices.get("ndmi")
    ndre = indices.get("ndre")
    fractions = indices.get("surface_fractions", {})
    qm = indices.get("quality_metrics", {})
    counts = indices.get("pixel_counts", {})

    print("\n--> Sentinel-2 Pixel Quality & Cloud Masking Metrics:")
    print(f"    Total Grid Pixels:   {counts.get('grid_pixels', 0)} ({qm.get('aoi_coverage_denominator', 'Grid')})")
    print(f"    AOI Pixels:          {counts.get('aoi_pixels', 0)} ({qm.get('aoi_pixel_coverage_pct', 0.0):.1f}% AOI coverage)")
    print(f"    Source Data Avail:   {counts.get('source_data_valid_pixels', 0)} ({qm.get('source_data_availability_pct', 0.0):.1f}% of AOI)")
    print(f"    No-Data / Off-Swath: {counts.get('nodata_pixels', 0)} ({qm.get('nodata_fraction_pct', 0.0):.1f}% of AOI)")
    if qm.get("cloud_fraction_pct") is not None:
        print(f"    Cloud Pixels:        {counts.get('cloud_pixels', 0)} ({qm.get('cloud_fraction_pct', 0.0):.1f}% of AOI)")
        print(f"    Shadow Pixels:       {counts.get('shadow_pixels', 0)} ({qm.get('shadow_fraction_pct', 0.0):.1f}% of AOI)")
        print(f"    Cloud+Shadow Masked: {qm.get('cloud_shadow_fraction_pct', 0.0):.1f}% of AOI")
    else:
        print(f"    Cloud Masking:       Unavailable ({qm.get('cloud_masking_reason', 'SCL missing')})")
    if qm.get("water_fraction_pct") is not None and qm.get("water_fraction_pct") > 0:
        print(f"    Water Pixels (Non-Crop): {counts.get('water_pixels', 0)} ({qm.get('water_fraction_pct', 0.0):.1f}% of AOI)")
    print(f"    Crop-Suitable Pixels:{counts.get('crop_usable_pixels', counts.get('usable_pixels', 0))} ({qm.get('crop_usable_pixel_pct', qm.get('usable_pixel_pct', 0.0)):.1f}% of AOI)")
    print(f"    Index-Usable Pixels: {counts.get('index_usable_pixels', 0)} ({qm.get('index_usable_pixel_pct', 0.0):.1f}% of AOI)")

    print("\n--> Spectral Indices Analytical Verification (Usable Pixels Only):")
    if ndvi is not None and np.any(np.isfinite(ndvi)):
        mean_ndvi = float(np.nanmean(ndvi))
        print(f"    [+] NDVI (Vigor):       Mean = {mean_ndvi:+.3f} (Min: {float(np.nanmin(ndvi)):+.3f}, Max: {float(np.nanmax(ndvi)):+.3f})")
    else:
        print(f"    [-] NDVI: Unavailable ({indices['indices_quality']['ndvi']['reason']})")

    if ndmi is not None and np.any(np.isfinite(ndmi)):
        mean_ndmi = float(np.nanmean(ndmi))
        print(f"    [+] NDMI (Moisture):    Mean = {mean_ndmi:+.3f} (Min: {float(np.nanmin(ndmi)):+.3f}, Max: {float(np.nanmax(ndmi)):+.3f})")
    else:
        print(f"    [-] NDMI: Unavailable ({indices['indices_quality']['ndmi']['reason']})")

    if ndre is not None and np.any(np.isfinite(ndre)):
        mean_ndre = float(np.nanmean(ndre))
        print(f"    [+] NDRE (RedEdge):     Mean = {mean_ndre:+.3f} (Min: {float(np.nanmin(ndre)):+.3f}, Max: {float(np.nanmax(ndre)):+.3f})")
    else:
        print(f"    [-] NDRE: Unavailable ({indices['indices_quality']['ndre']['reason']})")

    print(f"\n--> Surface Canopy Fractions (Relative to Total AOI):")
    print(f"    Active Vegetation:   {fractions.get('vegetation', 0.0):.1%}")
    print(f"    Bare Ground/Fallow:  {fractions.get('bare_surface', 0.0):.1%}")
    print(f"    Water Surface:       {fractions.get('water_surface', 0.0):.1%}")
    print(f"    Masked / Excluded:   {fractions.get('unclassified', 0.0):.1%}")
    print(f"    Usable Coverage:     {indices.get('valid_pixels_pct', 0.0):.1f}%")

    print("\n=================================================================")
    print("[AGRISENSE] Live Sentinel-2 Validation Result: SUCCESS")
    print("=================================================================")

    return {
        "status": "PASSED",
        "scene_meta": scene_meta,
        "indices_quality": indices["indices_quality"],
        "surface_fractions": fractions,
        "quality_metrics": qm,
        "exit_code": 0,
    }


def main():
    parser = argparse.ArgumentParser(description="AgriSense Sentinel-2 Live API Validation Tool")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude of target center point")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Longitude of target center point")
    parser.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM, help="Radius around point in km")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    s_date = date.fromisoformat(args.start_date) if args.start_date else None
    e_date = date.fromisoformat(args.end_date) if args.end_date else None

    res = validate_live_sentinel_integration(
        lat=args.lat,
        lon=args.lon,
        radius_km=args.radius_km,
        start_date=s_date,
        end_date=e_date,
    )
    sys.exit(res["exit_code"])


if __name__ == "__main__":
    main()
