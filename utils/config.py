"""
Central configuration for the app.

Loads API keys from .env and decides whether each external service should
run in MOCK MODE (no key present -> use realistic fake data so the whole
pipeline still runs end-to-end for demos) or LIVE MODE (real API calls).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys -----------------------------------------------------------
SENTINEL_HUB_CLIENT_ID = os.getenv("SENTINEL_HUB_CLIENT_ID", "")
SENTINEL_HUB_CLIENT_SECRET = os.getenv("SENTINEL_HUB_CLIENT_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Mode flags -----------------------------------------------------------
# TODO: insert API key — once these are set in .env, the flags below flip
# to False automatically and the real API calls in agents/*.py kick in.
SENTINEL_HUB_MOCK = not (SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET)
GEMINI_MOCK = not GEMINI_API_KEY


import sys


def get_sentinel_hub_credentials() -> tuple[str, str]:
    mod = sys.modules.get("utils.config")
    cid = getattr(mod, "SENTINEL_HUB_CLIENT_ID", SENTINEL_HUB_CLIENT_ID) or os.getenv("SENTINEL_HUB_CLIENT_ID", "")
    sec = getattr(mod, "SENTINEL_HUB_CLIENT_SECRET", SENTINEL_HUB_CLIENT_SECRET) or os.getenv("SENTINEL_HUB_CLIENT_SECRET", "")
    return cid, sec


def is_sentinel_hub_mock() -> bool:
    mod = sys.modules.get("utils.config")
    return getattr(mod, "SENTINEL_HUB_MOCK", SENTINEL_HUB_MOCK)


def set_sentinel_hub_credentials(client_id: str, client_secret: str) -> None:
    global SENTINEL_HUB_CLIENT_ID, SENTINEL_HUB_CLIENT_SECRET, SENTINEL_HUB_MOCK
    SENTINEL_HUB_CLIENT_ID = client_id
    SENTINEL_HUB_CLIENT_SECRET = client_secret
    SENTINEL_HUB_MOCK = not (bool(client_id) and bool(client_secret))
    os.environ["SENTINEL_HUB_CLIENT_ID"] = client_id
    os.environ["SENTINEL_HUB_CLIENT_SECRET"] = client_secret
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        cid_found, sec_found = False, False
        new_lines = []
        for line in lines:
            if line.startswith("SENTINEL_HUB_CLIENT_ID="):
                new_lines.append(f"SENTINEL_HUB_CLIENT_ID={client_id}\n")
                cid_found = True
            elif line.startswith("SENTINEL_HUB_CLIENT_SECRET="):
                new_lines.append(f"SENTINEL_HUB_CLIENT_SECRET={client_secret}\n")
                sec_found = True
            else:
                new_lines.append(line)
        if not cid_found:
            new_lines.append(f"SENTINEL_HUB_CLIENT_ID={client_id}\n")
        if not sec_found:
            new_lines.append(f"SENTINEL_HUB_CLIENT_SECRET={client_secret}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    load_dotenv(override=True)


def get_gemini_api_key() -> str:
    load_dotenv(override=True)
    return os.getenv("GEMINI_API_KEY", "")


def is_gemini_mock() -> bool:
    return not bool(get_gemini_api_key())


def set_gemini_api_key(key: str) -> None:
    os.environ["GEMINI_API_KEY"] = key
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        found = False
        new_lines = []
        for line in lines:
            if line.startswith("GEMINI_API_KEY="):
                new_lines.append(f"GEMINI_API_KEY={key}\n")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"GEMINI_API_KEY={key}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    load_dotenv(override=True)


# --- Geography ------------------------------------------------------------
# Rough bounding box for Maharashtra state (lon_min, lat_min, lon_max, lat_max)
MAHARASHTRA_BBOX = (72.6, 15.6, 80.9, 22.1)

# Max area (in km^2) a user can select in one query — keeps Sentinel Hub
# requests small and fast for a demo / free-tier account.
MAX_AREA_KM2 = 2500

# Max acceptable cloud cover percentage before the Input Agent triggers a
# re-query with a shifted date range.
MAX_CLOUD_COVER_PCT = 30

# Default date range width (days) to search for a clear-sky image.
DATE_RANGE_DAYS = 14

# Whether analysis is restricted strictly to Maharashtra bounding box or allows general AOIs.
RESTRICT_TO_MAHARASHTRA = os.getenv("RESTRICT_TO_MAHARASHTRA", "true").lower() in ("true", "1", "yes")

# Whether a failure in live satellite retrieval may fall back to synthetic mock imagery with explicit labeling,
# or immediately fail with a hard error.
ALLOW_LIVE_FALLBACK_TO_MOCK = os.getenv("ALLOW_LIVE_FALLBACK_TO_MOCK", "true").lower() in ("true", "1", "yes")

# --- Crop Identification & Agricultural Eligibility Thresholds ------------
# Maximum acceptable RMS distance between observed NDVI phenology series and
# reference signature for a confident crop identification. Above this, crop is 'uncertain'.
CROP_MAX_RMS_DISTANCE = 0.18

# Minimum score margin between best and second-best candidate crops.
# Below this margin, the temporal signature is ambiguous -> 'uncertain'.
CROP_MIN_CONFIDENCE_MARGIN = 0.06

# Minimum vegetation fraction (NDVI > 0.22) expected in an active agricultural canopy.
AGRICULTURAL_MIN_VEG_FRACTION = 0.10


def get_mode_summary() -> dict:
    """Returns the current data source and execution mode configuration."""
    live_sh = bool(SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET)
    live_gemini = not is_gemini_mock()
    return {
        "sentinel_hub_live": live_sh,
        "sentinel_hub_mode": "live" if live_sh else "mock",
        "gemini_live": live_gemini,
        "gemini_mode": "live" if live_gemini else "mock",
        "restrict_to_maharashtra": RESTRICT_TO_MAHARASHTRA,
        "allow_live_fallback": ALLOW_LIVE_FALLBACK_TO_MOCK,
    }
