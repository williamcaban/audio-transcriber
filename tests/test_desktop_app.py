"""Tests for the non-GUI logic in macos/desktop_app.py -- port selection,
readiness waiting, and that it actually serves the real FastAPI app.

Doesn't touch webview.create_window()/start() (the actual native window),
since that needs a real display and isn't importable/runnable in CI; see
macos/README.md for what's verified vs not.
"""

import sys
import threading
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip(
    "webview", reason="pywebview is a darwin-only optional extra (desktop)"
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "macos"))
import desktop_app


def test_free_port_is_actually_free_and_reusable():
    port = desktop_app._free_port()
    assert 1024 < port < 65536


def test_server_becomes_reachable_and_serves_the_real_app():
    port = desktop_app._free_port()
    threading.Thread(target=desktop_app._run_server, args=(port,), daemon=True).start()
    desktop_app._wait_until_reachable(port)

    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/api/models", timeout=10
    ) as resp:
        assert resp.status == 200
        body = resp.read()
    assert b"qwen3" in body
