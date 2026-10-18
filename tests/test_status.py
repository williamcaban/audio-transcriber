"""Unit tests for the backend load-status tracker -- fast, no model load."""

from app.asr import status


def test_unknown_key_defaults_to_not_loaded():
    assert status.get("never-touched")["state"] == "not_loaded"


def test_loading_then_ready():
    status.mark_loading("a")
    entry = status.get("a")
    assert entry["state"] == "loading"
    assert entry["error"] is None
    assert "loading_seconds" in entry

    status.mark_ready("a")
    entry = status.get("a")
    assert entry["state"] == "ready"
    assert "loading_seconds" not in entry


def test_loading_then_error_preserves_message():
    status.mark_loading("b")
    status.mark_error("b", "boom")
    entry = status.get("b")
    assert entry["state"] == "error"
    assert entry["error"] == "boom"


def test_get_never_leaks_internal_started_at():
    status.mark_loading("c")
    assert "started_at" not in status.get("c")
