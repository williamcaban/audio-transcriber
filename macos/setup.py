"""py2app build script -- macOS only, untested here (written and reviewed on
Linux; this repo has no macOS/Xcode environment to actually run py2app in).

Try it, and if the build fails, the error output is the fastest way to get
it fixed -- share it rather than fighting py2app options blind.

Usage (run from the repo root, on macOS):
    uv sync --extra desktop
    uv pip install py2app
    uv run python macos/setup.py py2app

Produces dist/Audio Transcriber.app.
"""

import sys
from pathlib import Path

from setuptools import setup

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

APP = ["desktop_app.py"]
DATA_FILES = [("static", [str(p) for p in (REPO_ROOT / "static").iterdir()])]
OPTIONS = {
    "argv_emulation": False,
    "packages": ["app"],
    "includes": ["uvicorn", "webview"],
    "plist": {
        "CFBundleName": "Audio Transcriber",
        "CFBundleDisplayName": "Audio Transcriber",
        "CFBundleIdentifier": "com.williamcaban.audio-transcriber",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
    },
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
