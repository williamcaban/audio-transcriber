"""py2app build script -- macOS only.

Usage (run from the repo root, on macOS):
    uv run --extra desktop python macos/setup.py py2app

Produces dist/Audio Transcriber.app.
"""

import os
import sys
from pathlib import Path

from py2app.build_app import py2app as _py2app
from setuptools import setup

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app import __version__

# py2app resolves `app=[...]` and writes build/dist output relative to the
# current working directory, not this script's own directory -- chdir here
# so `uv run python macos/setup.py py2app` from the repo root (as
# documented) still finds desktop_app.py and puts build/ and dist/ under
# macos/ rather than the repo root.
os.chdir(Path(__file__).resolve().parent)

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
        "CFBundleShortVersionString": __version__,
        "NSHighResolutionCapable": True,
    },
}


class py2app_cmd(_py2app):
    """Clears setuptools' auto-populated install_requires before py2app's
    own validation sees it.

    Modern setuptools copies this project's pyproject.toml
    [project.dependencies] onto the Distribution automatically, even though
    this setup.py never passes install_requires itself. py2app >=0.28.9
    unconditionally rejects any non-empty install_requires
    (py2app/build_app.py: "install_requires is no longer supported").
    Clearing it here is safe: the actual runtime dependencies are already
    installed via `uv sync --extra desktop` before this ever runs, so this
    isn't saying "no dependencies," just "don't let py2app's unrelated
    check see setuptools' derived value."
    """

    def finalize_options(self):
        self.distribution.install_requires = []
        super().finalize_options()


setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    cmdclass={"py2app": py2app_cmd},
)
