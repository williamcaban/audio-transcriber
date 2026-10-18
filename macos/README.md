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

## Package as a double-clickable .app (untested, try it)

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
- **Known, unresolved**: past those fixes, py2app's module-dependency
  scanner (`modulegraph`, AST-based) hit a `RecursionError` while walking
  this project's dependency tree -- almost certainly `torch`/`transformers`,
  whose generated/huge modules are a known pain point for py2app and
  PyInstaller alike. Not yet fixed. If you hit this, a next step worth
  trying is raising `sys.setrecursionlimit()` in `setup.py` before calling
  `setup()`, or excluding specific heavy submodules py2app doesn't need to
  trace fully.

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
