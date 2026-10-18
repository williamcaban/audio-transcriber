# Audio Transcriber

[![CI](https://github.com/williamcaban/audio-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/williamcaban/audio-transcriber/actions/workflows/ci.yml)
[![Release](https://github.com/williamcaban/audio-transcriber/actions/workflows/release.yml/badge.svg)](https://github.com/williamcaban/audio-transcriber/actions/workflows/release.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Small FastAPI app that transcribes audio and video files to Markdown or
Word, with a choice of two ASR backends (`app/asr/`):

| | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) |
|---|---|---|
| Timestamps | Native, word-level | Chunk-level only (~30s), no native per-word timestamps |
| Source language | Always auto-detects; **cannot be forced or reported** (no API for it) | Can force a language, and reports what it detected |
| Languages | 25 (European) | 30 (broader: adds Arabic, Chinese, Japanese, Korean, ...) |
| transformers release | Pre-release (installed from GitHub `main`) | Stable release |
| Chunk size used here | 10 minutes | 30 seconds (its audio encoder's native window) |

Neither supports **Guaraní** — that's a known gap, not a bug. If it becomes
a real need, that would mean wiring in a third model (e.g. Whisper
large-v3, which has weak but nonzero Guaraní coverage).

Pick Parakeet when you want precise word-level timestamps and don't care
about forcing a language (its own auto-detect is usually right anyway);
pick Qwen3-ASR when you need to force a language or want its detected-
language label, and coarser timestamps are fine. **Qwen3-ASR is the default**
(language control matters more in practice than word-level timestamps) --
switch to Parakeet in the Model dropdown if you want those instead.

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages the Python 3.13 install and the venv — no need to have 3.13 on PATH yourself)
- `ffmpeg` and `ffprobe` on PATH (audio/video extraction)
- A CUDA GPU is strongly recommended. It will run on CPU, but slowly, and
  long files may be impractical.

Uses the `transformers`-native path (`AutoModelForTDT`/`AutoProcessor`)
rather than the NeMo toolkit, specifically because NeMo's dependency chain
(pytorch-lightning, lhotse, megatron-core, ...) lags new Python releases by
months and this project targets 3.13+. The tradeoff: Parakeet TDT support in
`transformers` is very recent and not yet in a stable release, so
`pyproject.toml` installs it straight from the GitHub `main` branch — pin it
to a commit SHA (see the comment in that file) once you've confirmed a
version that works for you.

## Setup

```bash
uv sync
```

This creates `.venv` with Python 3.13 (fetched automatically by uv if not
already installed) and installs everything from `uv.lock`. If you need a
specific CUDA build of torch, install it first with
`uv pip install torch --index-url <your CUDA index>` before `uv sync`, or
add an explicit `[tool.uv.sources]` entry for `torch` in `pyproject.toml`
(see https://docs.astral.sh/uv/guides/integration/pytorch/).

## Run

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 (or `http://<host>:8000` if running on a
remote GPU box) and drop in a file. The model downloads from Hugging Face on
first startup (a few GB) and is cached by `huggingface_hub` after that.

## Running on macOS

Three options: run the web app natively, run it in a container, or build/
download an actual native window instead of a browser tab.

### Native (recommended on Apple Silicon)

```bash
brew install uv ffmpeg
git clone https://github.com/williamcaban/audio-transcriber.git
cd audio-transcriber
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`app/asr/common.py`'s `detect_device()` checks CUDA, then Apple's MPS
backend, then CPU (shared by both backends). Both models' `transformers`
ports are very new, so MPS op coverage isn't guaranteed — if
`model.generate()` errors on an unsupported op, force CPU by editing
`detect_device()` to always return `"cpu"` (slower, but guaranteed to
work), or set `PYTORCH_ENABLE_MPS_FALLBACK=1` to let unsupported ops
silently fall back to CPU per-op.

### Native window (a real `.app`, not a browser tab)

Confirmed working, both built locally and via CI on a real `macos-latest`
runner. Quickest way to try it without building anything:

```bash
# download Audio-Transcriber-macOS.zip from
# https://github.com/williamcaban/audio-transcriber/releases, then:
unzip Audio-Transcriber-macOS.zip
xattr -cr "Audio Transcriber.app"   # it's unsigned -- clears the quarantine flag
open "Audio Transcriber.app"
```

To build it yourself instead of downloading:

```bash
uv run --extra desktop python macos/setup.py py2app
```

Produces `macos/dist/Audio Transcriber.app`. See
[`macos/README.md`](macos/README.md) for the full build-from-source story
(including the real bugs hit and fixed along the way) and
[Releases and dev builds](#releases-and-dev-builds) below for where
pre-built copies come from. There's also a non-bundled way to get a native
window, useful while developing: `uv run --extra desktop python
macos/desktop_app.py` runs the server in-process and opens it in a
pywebview window directly, no packaging step.

### Container (Podman Desktop)

Podman on macOS runs containers inside a Linux VM, which has no access to
the host's GPU/Metal — this path is CPU-only regardless of your Mac's
hardware, so expect it to be noticeably slower than native.

```bash
brew install podman
podman machine init
podman machine start

# public, no login needed -- :dev tracks main (see Releases below for
# :latest/:vX.Y.Z once a version is tagged)
podman pull ghcr.io/williamcaban/audio-transcriber:dev
podman run --rm -p 8080:8080 \
  -v audio-transcriber-model-cache:/opt/app-root/src/.cache/huggingface \
  ghcr.io/williamcaban/audio-transcriber:dev
```

Then open http://localhost:8080 either way the image is pulled. See
[Running in a container](#running-in-a-container) below for more on the
model-cache volume and what the image contains.

## Running in a container

`Containerfile` builds on Fedora's official Python 3.14 application image
([quay.io/fedora/python-314](https://quay.io/repository/fedora/python-314))
plus `ffmpeg-free`. It installs dependencies with the base image's
preinstalled `pip`, from `requirements.txt` — a pinned export of `uv.lock`
so the container gets the exact versions you tested with locally under uv.
Regenerate it whenever `pyproject.toml`/`uv.lock` change:

```bash
uv export --no-hashes --no-dev --no-emit-project -o requirements.txt
```

```bash
podman build -t audio-transcriber:latest -f Containerfile .

# CPU-only run
podman run --rm -p 8080:8080 \
  -v audio-transcriber-model-cache:/opt/app-root/src/.cache/huggingface:Z \
  audio-transcriber:latest

# With an NVIDIA GPU passed through (requires the NVIDIA Container Toolkit's
# CDI setup: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
podman run --rm -p 8080:8080 --device nvidia.com/gpu=all \
  -v audio-transcriber-model-cache:/opt/app-root/src/.cache/huggingface:Z \
  audio-transcriber:latest
```

Then open http://localhost:8080. The named volume keeps the downloaded model
weights (a few GB) across container restarts/recreations — without it,
every fresh container re-downloads from Hugging Face on first request.

The base image is the SCLorg/OpenShift S2I Python image (runs as uid 1001,
`WORKDIR /opt/app-root/src`), so this also deploys cleanly on OpenShift with
its arbitrary-UID security model; swap the `FROM` line for
`quay.io/fedora/python-313` if you'd rather track 3.13 instead (local dev
via `uv` still targets 3.13 — see `.python-version` — `pyproject.toml` only
requires `>=3.13`, so either image works).

## Releases and dev builds

`.github/workflows/release.yml` builds and publishes both deployable
artifacts automatically -- no need to build either yourself unless you want
to:

- **Version tags** (`v0.1.0`, ...) publish a permanent
  [GitHub Release](https://github.com/williamcaban/audio-transcriber/releases)
  with `Audio-Transcriber-macOS.zip` attached, and push the container image
  as `ghcr.io/williamcaban/audio-transcriber:<tag>` plus `:latest`.
- **Every push to `main`** overwrites a single rolling
  [`dev` pre-release](https://github.com/williamcaban/audio-transcriber/releases/tag/dev)
  (same zipped `.app`) and the `:dev` container tag, with whatever's
  currently on `main` -- not a stable release, just the latest build without
  waiting for a version tag.

Both jobs run on a real target runner (`ubuntu-latest` for the container,
`macos-latest` for the `.app`), so a green badge above means the actual
artifacts build, not just that the Python import graph works.

## How it works

See [`docs/how-it-works.md`](docs/how-it-works.md) for the processing
pipeline (upload -> chunking -> per-backend transcription -> rendering),
how model loading/availability is surfaced to the UI, and how job
cancellation works.

## Language: one backend fakes it, the other doesn't

Parakeet TDT has no API to force a specific source language, and this
`transformers` port doesn't expose a detected-language field either
(confirmed by reading `processing_parakeet.py`/`generation_parakeet.py` in
the installed package -- there's no language token, prompt mechanism, or
output field anywhere). It's a pure auto-detecting acoustic model. That's
why there's no language picker for it in the UI: a dropdown that can't
actually change the model's behavior would just be a lie dressed as a
feature.

Qwen3-ASR is different: it's genuinely forced by prefilling the chat prompt
with `language <NAME><asr_text>` (`processing_qwen3_asr.py`), and reports
what it detected when you don't force one. Forcing the *wrong* language
doesn't translate the audio -- it just biases the transcription token
choices toward that language, which can degrade output if the audio is
clearly in a different one.

## Known limitations / next steps

- The transformers port of Parakeet TDT is pre-release; the exact shape of
  its per-token timestamp output isn't pinned down by stable docs, so
  `app/asr/parakeet.py`'s `_entries_to_segments` is deliberately defensive
  and falls back to one untimed block of text per chunk if parsing fails.
  If that fallback is firing in practice, re-check the model card for the
  current output format and tighten the parser.
- No persistent job history -- add a DB if you want transcripts to survive restarts.
- One model instance per backend; each processes one job at a time
  (background tasks run sequentially against whichever backend's already-
  loaded model). Fine for personal use, not for concurrent multi-user load.
  Using both backends in the same process run means both stay loaded in
  memory afterward -- there's no unloading.
- Long files are chunked at fixed boundaries with no overlap, so a word
  spanning a boundary can be slightly garbled. For Qwen3-ASR's 30s chunks
  this happens far more often than Parakeet's 10-minute ones.
- Qwen3-ASR has no native word-level timestamps; real ones would need a
  second model (`Qwen3-ForcedAligner-0.6B`, only 11 of its 30 languages) and
  a second inference pass. Not implemented -- each chunk is one timestamped
  segment instead.

## Development

See [`docs/development.md`](docs/development.md) for linting, running
tests, and how CI/release automation are split (fast lint+test on every PR
vs. the slower container/macOS artifact builds on push to `main`).

## License

[Apache License 2.0](LICENSE).
