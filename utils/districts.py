"""
Rough district lookup + crop-calendar constraints for Maharashtra.

In a full build, district boundaries and crop-calendar tables would come
straight from data.gov.in / the Maharashtra Dept. of Agriculture (they
publish district-wise crop area statistics and sowing calendars). The
bounding boxes and crop lists below are simplified stand-ins covering a
handful of major districts, good enough to (a) constrain which crops are
plausible for a given place/season, which meaningfully improves the
classifier's credibility, and (b) demonstrate the district-aggregation
feature end to end. Swap `DISTRICTS` for real polygon boundaries when
you have them.
"""
from shapely.geometry import shape, box, Point

# Approximate bounding boxes (lon_min, lat_min, lon_max, lat_max) — NOT
# survey-grade district boundaries, just enough to place a polygon in the
# right general district for a demo.
DISTRICTS = {
    "Nagpur": (78.8, 20.7, 79.6, 21.3),
    "Amravati": (77.2, 20.6, 78.0, 21.3),
    "Nashik": (73.5, 19.7, 74.4, 20.6),
    "Pune": (73.4, 17.9, 74.5, 19.2),
    "Solapur": (75.3, 17.3, 76.3, 18.4),
    "Chhatrapati Sambhajinagar": (75.0, 19.5, 75.9, 20.3),
    "Kolhapur": (73.8, 16.3, 74.5, 17.0),
    "Jalgaon": (75.0, 20.6, 76.1, 21.4),
}

# Which crops each district is best known for (used to re-rank/penalize
# implausible classifications), roughly reflecting real cropping patterns
# (e.g. Kolhapur/Solapur = sugar belt, Nagpur/Jalgaon = cotton belt).
DISTRICT_CROP_EMPHASIS = {
    "Nagpur": ["cotton", "tur", "soybean", "jowar"],
    "Amravati": ["cotton", "soybean", "tur"],
    "Nashik": ["soybean", "jowar", "sugarcane"],
    "Pune": ["sugarcane", "soybean", "jowar"],
    "Solapur": ["jowar", "sugarcane", "cotton"],
    "Chhatrapati Sambhajinagar": ["cotton", "jowar", "tur"],
    "Kolhapur": ["sugarcane", "soybean"],
    "Jalgaon": ["cotton", "jowar", "tur"],
}

# Simplified two-season crop calendar. Sugarcane is perennial so it's
# folded into both seasons' base lists.
CROP_CALENDAR = {
    "kharif": ["cotton", "soybean", "jowar", "tur", "sugarcane"],
    "rabi": ["jowar", "sugarcane"],
}


def season_for_date(d) -> str:
    """June-October ~ Kharif (monsoon-sown); rest of the year ~ Rabi."""
    return "kharif" if d.month in (6, 7, 8, 9, 10) else "rabi"


def locate_district(polygon_geojson: dict) -> str:
    """Returns the district whose bounding box contains the polygon's
    centroid, or the nearest district by centroid distance if it falls
    outside all of them, or 'Unknown' if the geometry can't be parsed."""
    try:
        centroid = shape(polygon_geojson).centroid
    except Exception:
        return "Unknown"

    for name, bbox in DISTRICTS.items():
        if box(*bbox).contains(centroid):
            return name

    # Not inside any mapped district's bbox — fall back to nearest by
    # centroid distance so we still get a plausible crop-calendar context.
    def _dist(bbox):
        minx, miny, maxx, maxy = bbox
        c = box(minx, miny, maxx, maxy).centroid
        return centroid.distance(c)

    nearest = min(DISTRICTS, key=lambda name: _dist(DISTRICTS[name]))
    return f"{nearest} (nearest match)"


def plausible_crops(district: str, season: str) -> list[str]:
    """Crops plausible for this district/season, ranked by local emphasis."""
    base = CROP_CALENDAR.get(season, [])
    clean_district = district.replace(" (nearest match)", "")
    emphasis = DISTRICT_CROP_EMPHASIS.get(clean_district)
    if emphasis:
        ranked = [c for c in emphasis if c in base]
        if ranked:
            return ranked
    return base
