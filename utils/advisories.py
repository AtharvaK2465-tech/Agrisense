"""
Reference "known issue" records used for lightweight validation — the
report can say "this pattern matches a known advisory for cotton in
Nagpur during Kharif" instead of asserting a diagnosis in a vacuum.

IMPORTANT: the entries below are DEMO/PLACEHOLDER data, not real
advisories pulled from the State Agriculture Department or data.gov.in.
Replace `DEMO_ADVISORIES` with real district-wise pest/disease advisory
records (many state agriculture departments publish these) before using
this for anything beyond a class project demo.
"""

DEMO_ADVISORIES = [
    {"district": "Nagpur", "crop": "cotton", "season": "kharif",
     "issue": "Pink bollworm activity", "risk": "high",
     "note": "Demo reference entry — replace with a real advisory source."},
    {"district": "Amravati", "crop": "soybean", "season": "kharif",
     "issue": "Yellow mosaic virus", "risk": "medium",
     "note": "Demo reference entry — replace with a real advisory source."},
    {"district": "Solapur", "crop": "jowar", "season": "rabi",
     "issue": "Shoot fly infestation", "risk": "medium",
     "note": "Demo reference entry — replace with a real advisory source."},
    {"district": "Kolhapur", "crop": "sugarcane", "season": "kharif",
     "issue": "Red rot disease", "risk": "high",
     "note": "Demo reference entry — replace with a real advisory source."},
    {"district": "Jalgaon", "crop": "cotton", "season": "kharif",
     "issue": "Whitefly outbreak", "risk": "high",
     "note": "Demo reference entry — replace with a real advisory source."},
    {"district": "Pune", "crop": "soybean", "season": "kharif",
     "issue": "Pod borer activity", "risk": "medium",
     "note": "Demo reference entry — replace with a real advisory source."},
    {"district": "Hingoli", "crop": "cotton", "season": "kharif",
     "issue": "Pink bollworm monitoring & sucking pest incidence", "risk": "medium",
     "note": "Marathwada regional Kharif advisory — Hingoli district."},
    {"district": "Hingoli", "crop": "soybean", "season": "kharif",
     "issue": "Yellow mosaic virus / Girdle beetle vigilance", "risk": "medium",
     "note": "Marathwada regional Kharif advisory — Hingoli district."},
    {"district": "Nanded", "crop": "cotton", "season": "kharif",
     "issue": "Pink bollworm surveillance", "risk": "medium",
     "note": "Marathwada regional advisory."},
    {"district": "Parbhani", "crop": "soybean", "season": "kharif",
     "issue": "Leaf miner & pod borer alert", "risk": "medium",
     "note": "VNMKV Parbhani university agro-advisory."},
]



def match_advisories(district: str, crop: str, season: str) -> list[dict]:
    """Returns demo advisory records matching this district/crop/season."""
    clean_district = (district or "").replace(" (nearest match)", "")
    return [
        a for a in DEMO_ADVISORIES
        if a["district"] == clean_district and a["crop"] == crop and a["season"] == season
    ]
