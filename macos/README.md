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
uv sync --extra desktop
uv run python macos/desktop_app.py
```

This opens a native window pointed at a locally-run copy of the server (a
random free port, not the usual 8000/8080). Model loading behaves exactly
like the browser version -- the "loading model..." banner and Model/Source
language dropdowns all work the same, since it's the same `static/
index.html` and `app/` code, just hosted in a window instead of a tab.

## Package as a double-clickable .app (untested, try it)

[py2app](https://py2app.readthedocs.io/) bundles a Python app into a
standalone macOS `.app`. It's a macOS-only build tool, so it's not in
`pyproject.toml`'s dependency groups (those get resolved on every
platform, even when unused) -- install it separately:

```bash
uv sync --extra desktop
uv pip install py2app
uv run python macos/setup.py py2app
```

This should produce `macos/dist/Audio Transcriber.app`. Known/likely rough
edges:

- **Fixed 2026-10-18**: `setup_requires=["py2app"]` triggered setuptools'
  legacy `fetch_build_eggs` auto-install path, which fails on current
  setuptools with `error: install_requires is no longer supported`. Removed
  it -- since these instructions already have you `uv pip install py2app`
  explicitly first, that mechanism was redundant anyway. If you hit this
  exact error, `git pull` for the fix.
- `packages: ["app"]` in `setup.py` tells py2app to bundle the `app/`
  package, but py2app's package discovery can be picky about what's
  importable from its own working directory vs the repo root -- if the
  build fails to find `app`, that's the first thing to check.
- The bundled app needs network access on first run to download the
  Qwen3-ASR/Parakeet model weights into the user's Hugging Face cache (same
  as any other first run) -- there's no model baked into the `.app` itself.
- `torch`/`transformers` are large; py2app may need `--includes` or
  `--packages` tweaks beyond what's in `setup.py` to pick up their full
  dependency trees. If the built `.app` crashes on launch with an
  `ImportError`, that's almost certainly this.

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
