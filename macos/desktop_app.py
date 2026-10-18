"""Desktop wrapper: runs the FastAPI server in-process and shows it in a
native window (WKWebView on macOS) via pywebview, instead of a browser tab.

This reuses the exact same app/ code as the web version -- nothing here
talks to the model directly, it just hosts the existing server locally and
points a native window at it.

Run with:
    uv sync --extra desktop
    uv run python macos/desktop_app.py

To package as a double-clickable .app, see setup.py in this directory.
"""

import socket
import sys
import threading
import time
from pathlib import Path

# This script's own directory (macos/) is what ends up on sys.path by
# default, not the repo root -- add the repo root explicitly so `from app
# import main` resolves regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
import webview

from app.main import app as fastapi_app

HOST = "127.0.0.1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def _run_server(port: int) -> None:
    uvicorn.run(fastapi_app, host=HOST, port=port, log_level="warning")


def _wait_until_reachable(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(
        f"Server did not start listening on port {port} within {timeout}s"
    )


def main() -> None:
    port = _free_port()
    # Daemon thread: when the window closes and webview.start() returns,
    # main() returns and the process exits, taking this thread down with
    # it -- no separate shutdown signal needed for a single-window desktop
    # app like this.
    threading.Thread(target=_run_server, args=(port,), daemon=True).start()
    _wait_until_reachable(port)

    webview.create_window(
        "Audio Transcriber",
        f"http://{HOST}:{port}",
        width=960,
        height=800,
        min_size=(640, 480),
    )
    webview.start()


if __name__ == "__main__":
    main()
