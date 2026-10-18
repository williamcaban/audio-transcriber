# macOS desktop wrapper

A thin native shell around the existing FastAPI app: it runs the same
server used by the browser version, in-process, and shows it in a native
window (WKWebView on macOS, via [pywebview](https://pywebview.flowrl.com/))
instead of a browser tab. No model code is duplicated or reimplemented --
this is a presentation layer on top of `app/`.

**Verification status:** `uv run python macos/desktop_app.py` is confirmed
working on real macOS (2026-10-18) -- the native WKWebView window comes up
and runs the app correctly. Built and checked from a Linux machine with no
macOS/Xcode access, so this was genuinely unverified until tried. The
py2app `.app` bundle below is still untested -- try it and report back what
breaks.

## Run without bundling

```bash
uv run --extra desktop python macos/desktop_app.py
```

One command: `--extra desktop` tells `uv run` to sync in `pywebview` (and,
for the packaging step below, `py2app`) for just this invocation, so there's
no separate `uv sync` step to remember.

This opens a native window pointed at a locally-run copy of the server (a
random free port, not the usual 8000/8080). Model loading behaves exactly
like the browser version -- the "loading model..." banner and Model/Source
language dropdowns all work the same, since it's the same `static/
index.html` and `app/` code, just hosted in a window instead of a tab.

## Package as a double-clickable .app

**Confirmed working** (2026-10-18): both a real local build on William's
Mac and the `macos-build` CI job (on a real `macos-latest` runner) produce
the `.app` successfully.

### Download a pre-built copy instead of building it

Every version is published to [GitHub Releases](https://github.com/williamcaban/audio-transcriber/releases):

- A tagged release (`v0.1.0`, ...) for each version.
- [`dev`](https://github.com/williamcaban/audio-transcriber/releases/tag/dev), a pre-release that's overwritten on every push to `main` -- the current state of development, not guaranteed stable.

Download `Audio-Transcriber-macOS.zip` from whichever release you want, unzip it, and you have `Audio Transcriber.app`.

(Every CI run also briefly keeps its own build as an Actions artifact, for
debugging a specific run -- see `.github/workflows/ci.yml`'s `macos-build`
job -- but Releases above are the actual place to get a copy from.)

**It's unsigned**, so macOS Gatekeeper will refuse to open it normally
("Audio Transcriber.app is damaged and can't be opened" or similar). Either
right-click -> Open (and confirm) the first time instead of double-clicking,
or clear the quarantine attribute a browser download adds:

```bash
xattr -cr "Audio Transcriber.app"
```

### Or build it yourself

[py2app](https://py2app.readthedocs.io/) bundles a Python app into a
standalone macOS `.app`:

```bash
uv run --extra desktop python macos/setup.py py2app
```

`py2app` is part of the `desktop` extra (darwin-only, like pywebview) --
**don't** `uv pip install py2app` as a separate step: `uv run` re-syncs the
environment to match the lockfile before every run, so anything installed
out-of-band that way gets silently removed the next time you run anything
with `uv run` (this bit a real build attempt: py2app installed and worked,
then `invalid command 'py2app'` on the very next `uv run` invocation). The
`--extra desktop` flag is what makes it persist correctly, in one command.

This should produce `macos/dist/Audio Transcriber.app`. Progress so far
(each found via real build attempts, not guessed):

- **Fixed**: `setup_requires=["py2app"]` triggered setuptools' legacy
  `fetch_build_eggs` auto-install path, which fails on current setuptools
  with `error: install_requires is no longer supported`. Removed it.
- **Fixed**: that same error came back even with py2app properly installed
  via the `desktop` extra -- turns out it's unrelated to `setup_requires`.
  Modern setuptools auto-copies this project's root `pyproject.toml`
  `[project.dependencies]` onto the build `Distribution`, and py2app
  `>=0.28.9` unconditionally rejects any non-empty `install_requires`
  (`py2app/build_app.py`, inside `finalize_options`). Fixed with a small
  `cmdclass` override in `setup.py` that clears it right before py2app's
  own check runs -- confirmed this specific error no longer occurs.
- **Fixed**: py2app resolves `app=["desktop_app.py"]` and writes `build/`
  and `dist/` relative to the *current working directory*, not this
  script's own directory -- following "run from the repo root" (as these
  instructions say) meant it looked for `desktop_app.py` at the repo root
  and failed. `setup.py` now `chdir`s to its own directory first.
- **Fixed**: confirmed on both Linux and real macOS -- py2app's
  module-dependency scanner (`modulegraph`, AST-based) hit a `RecursionError`
  walking this project's dependency tree, almost certainly `torch`/
  `transformers`' generated/huge modules (a known pain point for py2app and
  PyInstaller alike with large ML libraries). Fixed with
  `sys.setrecursionlimit(10000)` in `setup.py` before `setup()` runs --
  confirmed the scan then completes.
- **Fixed**: past the recursion fix, the build failed with `AttributeError:
  module 'zlib' has no attribute '__file__'`. py2app unconditionally does
  `import zlib; copy_file(zlib.__file__, ...)` while building the
  executable, assuming zlib is always a separate loadable extension file.
  uv-managed Python builds (python-build-standalone) statically link zlib
  into the interpreter for portability, so it has no `__file__`. Fixed by
  pointing `zlib.__file__` at the system's real zlib shared library
  (`ctypes.util.find_library("z")`) when it's missing -- harmless, since
  it's the same code, just not compiled into this interpreter as a
  standalone file.
- **Status after these fixes**: not yet confirmed to produce a working
  `.app` end to end -- py2app's static analysis of a `torch`/`transformers`
  dependency tree is known to be slow (a Linux-side verification run of the
  same fix ran 30+ minutes without finishing, almost certainly just slow
  rather than stuck, but wasn't worth waiting out on a platform that can't
  produce a real macOS bundle anyway). Try the actual build on macOS and
  give it time; report back whatever it does, success or a new error.

If you'd rather skip py2app entirely, a zero-code alternative for a
native-feeling window is Chrome/Safari's app mode pointed at a server you
start separately:

```bash
uv run uvicorn app.main:app --port 8000 &
open -a "Google Chrome" --args --app=http://localhost:8000
```

That gets you a chrome-less window with its own Dock icon in about one
command, at the cost of needing the server started separately rather than
bundled.
