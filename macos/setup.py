"""py2app build script -- macOS only.

Usage (run from the repo root, on macOS):
    uv run --extra desktop python macos/setup.py py2app

Produces dist/Audio Transcriber.app.
"""

import os
import subprocess
import sys
from pathlib import Path

from py2app.build_app import py2app as _py2app
from setuptools import setup

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app import __version__

# py2app's modulegraph scans every reachable module's AST recursively
# (ast.NodeVisitor), and torch/transformers include some very large and/or
# deeply nested generated files that blow past Python's default recursion
# limit (1000) partway through the scan -- not an infinite loop, just a
# deeper tree than the default allows. Confirmed: raising this is what gets
# the scan to complete.
sys.setrecursionlimit(10000)

# py2app unconditionally does `import zlib; copy_file(zlib.__file__, ...)`
# while building the executable, assuming zlib is always a separate loadable
# extension file. uv-managed Python builds (python-build-standalone)
# statically link zlib into the interpreter for portability, so it has no
# __file__ to begin with.
#
# Pointing zlib.__file__ at the "system" path (e.g. via
# ctypes.util.find_library("z") -> /usr/lib/libz.dylib) isn't enough on
# real macOS (Big Sur+): system libraries no longer exist as real files on
# disk at all, only inside the dyld shared cache (at most the SDK ships a
# text-based .tbd *link-time* stub, which isn't a copyable file either) --
# py2app's copy then fails with a plain FileNotFoundError. Confirmed:
# this is why the CI macos-latest runner's build succeeded (its image still
# has that path as a real file) while a real, current macOS machine's did
# not.
#
# Fix: compile a tiny real dylib that links against system libz -- the
# compiler resolves `-lz` fine via the SDK's .tbd stub, and the result is a
# genuine on-disk Mach-O file py2app CAN copy. Its contents don't matter at
# runtime: this interpreter's zlib support is already statically linked in,
# so this bundled copy is never actually loaded.
import zlib

if not hasattr(zlib, "__file__"):
    _stub_dir = Path(__file__).resolve().parent / "build"
    _stub_dir.mkdir(parents=True, exist_ok=True)
    _stub_path = _stub_dir / "libz_stub.dylib"
    if not _stub_path.exists():
        subprocess.run(["cc", "-dynamiclib", "-lz", "-o", str(_stub_path)], check=True)
    zlib.__file__ = str(_stub_path)

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
    # By default py2app zip-compresses pure-Python packages into
    # python313.zip for compactness. That breaks any package that ships a
    # native .dylib/.so as package *data* and dlopen()s it via a relative
    # path at runtime (rather than importing it as a normal compiled Python
    # extension module, which py2app's dependency scanner does handle
    # correctly) -- the dynamic linker can't load a library from inside a
    # zip archive.
    #
    # `soundfile` itself is a flat top-level module (soundfile.py), not a
    # directory package, so listing it here does nothing -- py2app's
    # "packages" mechanism only un-zips directory-style packages. The actual
    # bundled dylib lookup goes through a *separate* package, `_soundfile_
    # data` (soundfile.py does `import _soundfile_data; dirname(_soundfile_
    # data.__file__)`), confirmed via a real build+run attempt (OSError
    # loading its bundled libsndfile_arm64.dylib from inside python313.zip
    # even with `soundfile` listed here). `_soundfile_data` is what needs to
    # stay unzipped. `torch` and `numpy` are added preemptively, not
    # reactively: both are well-known to hit this exact failure mode in
    # py2app/PyInstaller builds for the same reason (bundled native
    # libraries loaded via relative paths, and both ARE real directory
    # packages so this option actually applies to them), so it's worth
    # fixing before the next build attempt rather than making William
    # iterate through each one individually. If another package hits the
    # same `OSError: cannot load library` pattern, add it here too (check
    # first whether it's a directory package -- if it's a flat module like
    # soundfile.py, find the actual package holding the bundled data).
    "packages": ["app", "_soundfile_data", "torch", "numpy"],
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
