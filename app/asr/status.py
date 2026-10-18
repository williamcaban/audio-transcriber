"""Tracks each backend's load state so the UI can show real availability
instead of the server hanging silently while a multi-GB model downloads.

Not byte-level download progress -- huggingface_hub doesn't expose that
through a simple public callback, and faking a percentage would be worse
than not showing one. This tracks state (not_loaded/loading/ready/error)
plus how long loading has taken, which is enough to tell "it's working" from
"it's stuck" without pretending to a precision we don't have.
"""

import threading
import time

_lock = threading.Lock()
_status: dict[str, dict] = {}


def _entry(key: str) -> dict:
    return _status.setdefault(
        key, {"state": "not_loaded", "started_at": None, "error": None}
    )


def get(key: str) -> dict:
    with _lock:
        entry = dict(_entry(key))
    started_at = entry.pop("started_at")
    if entry["state"] == "loading" and started_at is not None:
        entry["loading_seconds"] = time.time() - started_at
    return entry


def mark_loading(key: str) -> None:
    with _lock:
        _entry(key).update(state="loading", started_at=time.time(), error=None)


def mark_ready(key: str) -> None:
    with _lock:
        _entry(key).update(state="ready", error=None)


def mark_error(key: str, error: str) -> None:
    with _lock:
        _entry(key).update(state="error", error=error)
