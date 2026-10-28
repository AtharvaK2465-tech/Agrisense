"""
Geometry and AOI helpers for validating, parsing, and measuring user-drawn
or provided areas of interest before submission to satellite APIs.
"""
from typing import Any, Tuple, Optional
from shapely.geometry import shape, box, Point, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
from shapely import wkt

import utils.config as config
from utils.config import MAHARASHTRA_BBOX, MAX_AREA_KM2

# Rough degrees-to-km conversion at Maharashtra's latitude (~19°N).
_KM_PER_DEG_LAT = 111.0
_KM_PER_DEG_LON = 105.0  # cos(19°) * 111


def polygon_from_geojson(geojson: Any) -> BaseGeometry:
    """Convert a GeoJSON geometry/feature dict or WKT string into a shapely geometry.
    Handles GeoJSON Feature, FeatureCollection, Polygon, MultiPolygon, and WKT strings.
    """
    if isinstance(geojson, str):
        text = geojson.strip()
        if text.startswith("{"):
            import json
            geojson = json.loads(text)
        else:
            return wkt.loads(text)

    if isinstance(geojson, dict):
        if geojson.get("type") == "Feature":
            geojson = geojson.get("geometry", {})
        elif geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
            if not features:
                raise ValueError("FeatureCollection contains no features.")
            geojson = features[0].get("geometry", {})

        return shape(geojson)

    raise ValueError(f"Unsupported geometry type: {type(geojson)}")


def point_radius_to_geojson(lat: float, lon: float, radius_km: float = 1.0, as_circle: bool = True) -> dict:
    """
    Generates an AOI geometry from a center point and radius in kilometers.
    
    If as_circle=True (default):
      Constructs a true metric circular polygon using an Azimuthal Equidistant (AEQD)
      projection centered at (lat, lon) with metric radius (radius_km * 1000.0 m)
      and 32 quad segments (128 vertices), then projects back to WGS84 (EPSG:4326).
      This avoids degree-buffering distortion and accurately models a circular field parcel.
    If as_circle=False:
      Constructs the axis-aligned bounding box enclosing the circular buffer.
    """
    from shapely.geometry import mapping
    if as_circle:
        try:
            import pyproj
            from shapely.geometry import Point
            from shapely.ops import transform
            aeqd = pyproj.CRS(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
            wgs84 = pyproj.CRS("EPSG:4326")
            project_to_wgs84 = pyproj.Transformer.from_crs(aeqd, wgs84, always_xy=True).transform
            circle_m = Point(0, 0).buffer(radius_km * 1000.0, quad_segs=32)
            circle_geo = transform(project_to_wgs84, circle_m)
            return mapping(circle_geo)
        except Exception:
            pass  # Fall back to box if projection fails

    delta_lat = radius_km / _KM_PER_DEG_LAT
    delta_lon = radius_km / _KM_PER_DEG_LON
    poly = box(lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat)
    return mapping(poly)


def approx_area_km2(geom: BaseGeometry) -> float:
    """
    Calculates accurate geodesic area in square kilometers on the WGS84 ellipsoid.
    Falls back to spherical/cosine approximation if pyproj is unavailable.
    """
    if geom.is_empty:
        return 0.0
    try:
        import pyproj
        geod = pyproj.Geod(ellps="WGS84")
        area, _ = geod.geometry_area_perimeter(geom)
        return abs(float(area)) / 1e6
    except Exception:
        minx, miny, maxx, maxy = geom.bounds
        mean_lat = (miny + maxy) / 2.0
        import math
        km_per_deg_lon = math.cos(math.radians(mean_lat)) * 111.32
        deg_area = geom.area  # in square degrees
        km2 = deg_area * _KM_PER_DEG_LAT * km_per_deg_lon
        return abs(float(km2))


def validate_polygon(geojson: Any) -> Tuple[bool, str]:
    """
    Returns (is_valid, message). Rigorously checks:
      1. Geometry parsing (GeoJSON or WKT)
      2. Valid coordinate range (WGS84 lon [-180, 180], lat [-90, 90])
      3. Geometry validity and self-intersection repair
      4. State bounding constraint (if RESTRICT_TO_MAHARASHTRA is enabled)
      5. Configurable AOI area limit
    """
    try:
        geom = polygon_from_geojson(geojson)
    except Exception as e:
        return False, f"Could not parse the drawn shape: {e}"

    if geom.is_empty:
        return False, "The drawn shape is empty. Please draw a closed boundary."

    if not geom.is_valid:
        # Attempt safe topological repair
        try:
            geom = make_valid(geom)
        except Exception:
            return False, "The drawn shape has invalid geometry or unresolvable self-intersections."

    minx, miny, maxx, maxy = geom.bounds
    if not (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0):
        return False, f"Coordinates outside valid WGS-84 geographic bounds: ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})."

    if config.RESTRICT_TO_MAHARASHTRA:
        maha_box = box(*MAHARASHTRA_BBOX)
        if not geom.intersects(maha_box):
            return False, (
                "Selected area is outside Maharashtra (lon 72.6°-80.9°E, lat 15.6°-22.1°N). "
                "Current product setting restricts queries to Maharashtra. "
                "Set RESTRICT_TO_MAHARASHTRA=false to analyze any global AOI."
            )

    area = approx_area_km2(geom)
    if area < 0.0001:
        return False, "Selected area is too small (< 0.01 hectares). Please select a field parcel."

    if area > MAX_AREA_KM2:
        return False, (
            f"Selected area (~{area:,.1f} km²) exceeds the {MAX_AREA_KM2:,} km² limit "
            "for a single query. Draw a smaller region."
        )

    return True, f"Valid AOI selection (~{area:,.2f} km² / {area * 100:,.1f} ha)."


def get_aoi_metadata(geojson: Any) -> dict:
    """Extracts bounding box, centroid, and area statistics for display and provenance."""
    geom = polygon_from_geojson(geojson)
    if not geom.is_valid:
        geom = make_valid(geom)
    minx, miny, maxx, maxy = geom.bounds
    centroid = geom.centroid
    area_km2 = approx_area_km2(geom)
    return {
        "bounds": (round(minx, 5), round(miny, 5), round(maxx, 5), round(maxy, 5)),
        "centroid_lat": round(centroid.y, 5),
        "centroid_lon": round(centroid.x, 5),
        "area_km2": round(area_km2, 3),
        "area_hectares": round(area_km2 * 100.0, 2),
    }
