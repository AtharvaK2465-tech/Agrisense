"""
District lookup and crop-calendar constraints for Maharashtra (all 36 districts).

Combines:
1. Online reverse geocoding via OpenStreetMap Nominatim for exact real-world district
   boundaries (with in-memory cache and 2.5s timeout).
2. Complete offline bounding-box coverage for all 36 Maharashtra districts.
"""
import re
import requests
from shapely.geometry import shape, box, Point

# Cache to avoid repeated network calls for the same area
_GEOCODE_CACHE: dict[tuple[float, float], str] = {}

# Approximate bounding boxes (lon_min, lat_min, lon_max, lat_max) for all 36 districts
DISTRICTS = {
    # Marathwada
    "Hingoli": (76.6, 19.3, 77.6, 20.1),
    "Nanded": (76.9, 18.3, 78.3, 19.6),
    "Parbhani": (76.2, 18.9, 77.2, 20.0),
    "Jalna": (75.6, 19.3, 76.6, 20.5),
    "Chhatrapati Sambhajinagar": (74.6, 19.3, 76.1, 20.6),
    "Beed": (74.9, 18.5, 76.4, 19.4),
    "Latur": (76.2, 17.9, 77.3, 18.8),
    "Dharashiv": (75.2, 17.6, 76.5, 18.7),

    # Vidarbha
    "Nagpur": (78.5, 20.6, 79.6, 21.7),
    "Amravati": (76.6, 20.5, 78.5, 21.8),
    "Akola": (76.6, 20.2, 77.4, 21.3),
    "Buldhana": (75.9, 19.9, 76.9, 21.3),
    "Washim": (76.6, 19.9, 77.7, 20.8),
    "Yavatmal": (77.3, 19.4, 79.2, 20.7),
    "Wardha": (78.0, 20.3, 79.2, 21.4),
    "Chandrapur": (78.8, 19.5, 80.0, 20.9),
    "Gadchiroli": (79.7, 18.7, 80.9, 20.9),
    "Bhandara": (79.4, 20.8, 80.1, 21.6),
    "Gondia": (79.8, 20.6, 80.7, 21.7),

    # Khandesh & North Maharashtra
    "Nashik": (73.3, 19.5, 74.9, 20.9),
    "Dhule": (74.2, 20.6, 75.3, 21.6),
    "Nandurbar": (73.7, 21.0, 74.8, 22.0),
    "Jalgaon": (74.7, 20.2, 76.4, 21.4),
    "Ahmednagar": (73.9, 18.3, 75.6, 19.9),

    # Western Maharashtra
    "Pune": (73.3, 17.9, 75.2, 19.4),
    "Satara": (73.5, 17.1, 74.9, 18.2),
    "Solapur": (74.7, 17.1, 76.4, 18.5),
    "Sangli": (73.7, 16.7, 75.7, 17.6),
    "Kolhapur": (73.6, 15.7, 74.7, 17.2),

    # Konkan
    "Palghar": (72.6, 19.4, 73.3, 20.4),
    "Thane": (72.7, 19.0, 73.6, 19.8),
    "Mumbai City": (72.8, 18.9, 72.9, 19.0),
    "Mumbai Suburban": (72.7, 19.0, 73.0, 19.3),
    "Raigad": (72.8, 17.8, 73.6, 19.0),
    "Ratnagiri": (73.1, 16.5, 73.9, 18.1),
    "Sindhudurg": (73.3, 15.6, 74.2, 16.5),
}

# District aliases for geocoding normalization
DISTRICT_ALIASES = {
    "aurangabad": "Chhatrapati Sambhajinagar",
    "chhatrapati sambhaji nagar": "Chhatrapati Sambhajinagar",
    "osmanabad": "Dharashiv",
    "ahmednagar": "Ahmednagar",
    "ahilyanagar": "Ahmednagar",
}

DISTRICT_CROP_EMPHASIS = {
    "Hingoli": ["soybean", "cotton", "tur", "jowar"],
    "Nanded": ["cotton", "soybean", "tur", "jowar"],
    "Parbhani": ["cotton", "soybean", "tur", "jowar"],
    "Jalna": ["cotton", "soybean", "tur", "jowar"],
    "Chhatrapati Sambhajinagar": ["cotton", "jowar", "tur", "soybean"],
    "Beed": ["cotton", "soybean", "tur", "jowar"],
    "Latur": ["soybean", "tur", "jowar", "sugarcane"],
    "Dharashiv": ["soybean", "jowar", "tur", "sugarcane"],
    "Nagpur": ["cotton", "tur", "soybean", "jowar"],
    "Amravati": ["cotton", "soybean", "tur"],
    "Akola": ["cotton", "soybean", "tur"],
    "Buldhana": ["cotton", "soybean", "jowar", "tur"],
    "Washim": ["soybean", "cotton", "tur"],
    "Yavatmal": ["cotton", "soybean", "tur", "jowar"],
    "Wardha": ["cotton", "soybean", "tur"],
    "Chandrapur": ["cotton", "soybean", "tur"],
    "Gadchiroli": ["cotton", "soybean"],
    "Bhandara": ["soybean", "tur"],
    "Gondia": ["soybean", "tur"],
    "Nashik": ["soybean", "jowar", "sugarcane"],
    "Dhule": ["cotton", "jowar", "soybean"],
    "Nandurbar": ["cotton", "jowar", "soybean"],
    "Jalgaon": ["cotton", "jowar", "tur"],
    "Ahmednagar": ["sugarcane", "jowar", "soybean", "cotton"],
    "Pune": ["sugarcane", "soybean", "jowar"],
    "Satara": ["sugarcane", "soybean", "jowar"],
    "Solapur": ["jowar", "sugarcane", "cotton"],
    "Sangli": ["sugarcane", "jowar", "soybean"],
    "Kolhapur": ["sugarcane", "soybean"],
    "Palghar": ["soybean"],
    "Thane": ["soybean"],
    "Raigad": ["soybean"],
    "Ratnagiri": ["soybean"],
    "Sindhudurg": ["soybean"],
}

CROP_CALENDAR = {
    "kharif": ["cotton", "soybean", "jowar", "tur", "sugarcane"],
    "rabi": ["jowar", "sugarcane"],
}


def season_for_date(d) -> str:
    """June-October ~ Kharif (monsoon-sown); rest of the year ~ Rabi."""
    return "kharif" if d.month in (6, 7, 8, 9, 10) else "rabi"


def _normalize_district_name(raw_name: str) -> str | None:
    if not raw_name:
        return None
    cleaned = re.sub(r"\b(district|division|subdivision)\b", "", raw_name, flags=re.IGNORECASE).strip()
    lowered = cleaned.lower()
    if lowered in DISTRICT_ALIASES:
        return DISTRICT_ALIASES[lowered]
    for dname in DISTRICTS:
        if dname.lower() == lowered:
            return dname
    return None


def _reverse_geocode_district(lat: float, lon: float) -> str | None:
    cache_key = (round(lat, 3), round(lon, 3))
    if cache_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cache_key]

    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lon, "format": "json", "zoom": 10}
        headers = {"User-Agent": "AgriSenseCropMonitor/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=2.5)
        if r.status_code == 200:
            data = r.json()
            addr = data.get("address", {})
            for field in ["state_district", "county", "district", "city"]:
                val = addr.get(field)
                if val:
                    normalized = _normalize_district_name(val)
                    if normalized:
                        _GEOCODE_CACHE[cache_key] = normalized
                        return normalized
    except Exception:
        pass
    return None


def locate_district(polygon_geojson: dict) -> str:
    """Returns the precise district name for a polygon in Maharashtra.
    
    1. Tries high-precision reverse geocoding via OpenStreetMap.
    2. Falls back to Maharashtra 36-district bounding box containment.
    3. Falls back to closest centroid among all 36 districts.
    """
    try:
        poly_shape = shape(polygon_geojson)
        centroid = poly_shape.centroid
    except Exception:
        return "Unknown"

    # 1. Try real-world reverse geocode lookup first
    real_district = _reverse_geocode_district(centroid.y, centroid.x)
    if real_district:
        return real_district

    # 2. Check bounding box containment among 36 districts
    matching = []
    for name, bbox in DISTRICTS.items():
        if box(*bbox).contains(centroid):
            matching.append(name)

    if len(matching) == 1:
        return matching[0]
    elif len(matching) > 1:
        # If overlapping boxes, pick the one with closest centroid
        def _dist_to_center(dname):
            minx, miny, maxx, maxy = DISTRICTS[dname]
            c = box(minx, miny, maxx, maxy).centroid
            return centroid.distance(c)

        return min(matching, key=_dist_to_center)

    # 3. Outside all boxes: find closest district
    def _dist(bbox):
        minx, miny, maxx, maxy = bbox
        c = box(minx, miny, maxx, maxy).centroid
        return centroid.distance(c)

    nearest = min(DISTRICTS, key=lambda name: _dist(DISTRICTS[name]))
    return nearest


_GROUND_CONTEXT_CACHE: dict[tuple[float, float], dict] = {}


def detect_ground_context(polygon_geojson: dict, mean_ndvi: float | None = None) -> dict:
    """Identifies whether an area of interest is non-agricultural / built-up / bare ground
    (such as a college campus ground like VIT Pune, residential colonies, industrial zones,
    sports grounds, or uncultivated barren earth)."""
    try:
        poly_shape = shape(polygon_geojson)
        centroid = poly_shape.centroid
        lat, lon = centroid.y, centroid.x
    except Exception:
        return {"is_non_agri": False, "category": "Unknown", "reason": "Could not parse geometry"}

    # 1. Direct geographic coordinates for known non-agricultural campuses & zones:
    # A. Vishwakarma Institute of Technology (VIT Pune) Bibwewadi campus & sports ground
    if (18.455 <= lat <= 18.475) and (73.860 <= lon <= 73.878):
        return {
            "is_non_agri": True,
            "category": "Campus Ground (VIT Pune)",
            "reason": "Vishwakarma Institute of Technology campus ground / educational premises in Bibwewadi, Pune.",
        }

    # B. VIT Kondhwa Campus Ground
    if (18.450 <= lat <= 18.465) and (73.885 <= lon <= 73.900):
        return {
            "is_non_agri": True,
            "category": "Campus Ground (VIT Kondhwa)",
            "reason": "Vishwakarma Institute of Information Technology / VIT Kondhwa campus ground, Pune.",
        }

    # C. Pune Municipal Corporation (PMC) Dense City Core (Kothrud, Bibwewadi, Swargate, Shivajinagar, Camp)
    # Inside this dense municipal core, open grounds are college grounds, sports turf, or residential plots, not farmland.
    if (18.475 <= lat <= 18.555) and (73.805 <= lon <= 73.905):
        return {
            "is_non_agri": True,
            "category": "Urban Residential / City Area",
            "reason": "Dense Pune municipal corporation urban area (residential colonies, commercial zones, and urban grounds).",
        }

    # D. Bhosari / Pimpri MIDC Industrial Belt
    if (18.610 <= lat <= 18.665) and (73.780 <= lon <= 73.860):
        return {
            "is_non_agri": True,
            "category": "Industrial Zone (MIDC)",
            "reason": "Pimpri-Chinchwad / Bhosari industrial and manufacturing estate.",
        }

    # 2. Check cached or real-time OSM reverse geocode land-use / amenity
    cache_key = (round(lat, 3), round(lon, 3))
    if cache_key in _GROUND_CONTEXT_CACHE:
        return _GROUND_CONTEXT_CACHE[cache_key]

    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lon, "format": "json", "zoom": 17, "addressdetails": 1}
        headers = {"User-Agent": "AgriSenseCropMonitor/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=2.0)
        if r.status_code == 200:
            data = r.json()
            osm_class = str(data.get("class", "")).lower()
            osm_type = str(data.get("type", "")).lower()
            addr = data.get("address", {})
            display_name = str(data.get("display_name", "")).lower()

            # Check explicit non-agri keywords in display name
            keywords = [
                "institute of technology", "vit", "campus", "university", "college",
                "stadium", "sports ground", "cricket ground", "football ground",
                "midc", "industrial estate", "industrial area", "technology park", "it park",
                "residential society", "housing society", "apartments"
            ]
            for kw in keywords:
                if kw in display_name:
                    res = {
                        "is_non_agri": True,
                        "category": f"Non-Agricultural Land ({kw.title()})",
                        "reason": f"Identified as {kw} ({display_name[:70]}...).",
                    }
                    _GROUND_CONTEXT_CACHE[cache_key] = res
                    return res

            # Check explicit OSM non-agricultural classes
            if osm_class == "amenity" and osm_type in ["university", "college", "school", "hospital", "stadium", "parking"]:
                res = {
                    "is_non_agri": True,
                    "category": f"Institutional / Amenity Ground ({osm_type.title()})",
                    "reason": f"OpenStreetMap classified as {osm_class}:{osm_type}.",
                }
                _GROUND_CONTEXT_CACHE[cache_key] = res
                return res

            if osm_class == "leisure" and osm_type in ["pitch", "stadium", "sports_centre", "playground", "park", "track"]:
                res = {
                    "is_non_agri": True,
                    "category": f"Sports / Recreation Ground ({osm_type.title()})",
                    "reason": f"OpenStreetMap classified as sports ground ({osm_type}).",
                }
                _GROUND_CONTEXT_CACHE[cache_key] = res
                return res

            if osm_class == "landuse" and osm_type in ["industrial", "commercial", "retail", "construction", "residential", "cemetery", "railway"]:
                res = {
                    "is_non_agri": True,
                    "category": f"Urban Landuse ({osm_type.title()})",
                    "reason": f"OpenStreetMap classified as {osm_type} zone.",
                }
                _GROUND_CONTEXT_CACHE[cache_key] = res
                return res

            # Water bodies (lakes, reservoirs, rivers, wetlands)
            is_water_osm = (
                osm_class in ["waterway", "reservoir"]
                or (osm_class == "natural" and osm_type in ["water", "wetland", "bay"])
                or osm_type in ["water", "reservoir", "lake", "river", "canal"]
                or any(w in display_name for w in ["lake", "reservoir", "river", "dam", "wetland", "water body"])
            )
            if is_water_osm:
                res = {
                    "is_non_agri": True,
                    "category": "Water Body",
                    "reason": f"OpenStreetMap identified as water surface ({osm_class}:{osm_type}).",
                }
                _GROUND_CONTEXT_CACHE[cache_key] = res
                return res

            # Transport infrastructure (airports, major highways, rail yards)
            if osm_class in ["aeroway", "railway"] or osm_type in ["motorway", "trunk", "runway", "apron"]:
                res = {
                    "is_non_agri": True,
                    "category": "Transport / Infrastructure",
                    "reason": f"OpenStreetMap identified as transportation infrastructure ({osm_type}).",
                }
                _GROUND_CONTEXT_CACHE[cache_key] = res
                return res

            # Rural villages / hamlets are definitely agricultural
            if "village" in addr or "hamlet" in addr:
                res = {
                    "is_non_agri": False,
                    "category": "Agricultural Field",
                    "reason": f"Rural agricultural land in village {addr.get('village') or addr.get('hamlet')}.",
                }
                _GROUND_CONTEXT_CACHE[cache_key] = res
                return res
    except Exception:
        pass

    res = {
        "is_non_agri": False,
        "category": "Agricultural Field",
        "reason": "Normal agricultural terrain with active or seasonal vegetation.",
    }
    _GROUND_CONTEXT_CACHE[cache_key] = res
    return res


def plausible_crops(district: str, season: str) -> list[str]:
    """Crops plausible for this district/season, ranked by local emphasis."""
    base = CROP_CALENDAR.get(season, []).copy()
    # Non-agricultural ground and fallow are always plausible everywhere in all seasons
    for non_crop in ["no_crop", "fallow"]:
        if non_crop not in base:
            base.append(non_crop)

    clean_district = (district or "").replace(" (nearest match)", "").strip()
    emphasis = DISTRICT_CROP_EMPHASIS.get(clean_district)
    if emphasis:
        ranked = [c for c in emphasis if c in base]
        for non_crop in ["no_crop", "fallow"]:
            if non_crop not in ranked:
                ranked.append(non_crop)
        if ranked:
            return ranked
    return base

