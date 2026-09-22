"""
Tiny JSON-file "database" so past runs are browsable across a Streamlit
refresh (Streamlit's own session_state is wiped on reload). Good enough
for a single-user student project; swap for a real DB for multi-user use.
"""
import json
import os
from datetime import datetime

HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "history.json")
MAX_ENTRIES = 50


def load_history() -> list[dict]:
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_run(record: dict) -> list[dict]:
    """Appends a run summary (small dict of primitives only — no numpy
    arrays) to history.json, capped at MAX_ENTRIES, newest first."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    history = load_history()
    record = {**record, "saved_at": datetime.now().isoformat(timespec="seconds")}
    history.insert(0, record)
    history = history[:MAX_ENTRIES]
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    return history
