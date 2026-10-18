# Audio Transcriber

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

Two options to run the web app; native is recommended on Apple Silicon
since it can use the GPU via Metal (MPS), which a container can't reach.
For an actual native window instead of a browser tab, see
[`macos/README.md`](macos/README.md) -- a thin wrapper that reuses this
same server and UI, confirmed working on real macOS.

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

### Container (Podman Desktop)

Podman on macOS runs containers inside a Linux VM, which has no access to
the host's GPU/Metal — this path is CPU-only regardless of your Mac's
hardware, so expect it to be noticeably slower than native.

```bash
brew install podman
podman machine init
podman machine start

# the image is currently private — log in first
echo "<a GitHub token with read:packages>" | podman login ghcr.io -u <your-github-username> --password-stdin

podman pull ghcr.io/williamcaban/audio-transcriber:latest
podman run --rm -p 8080:8080 \
  -v audio-transcriber-model-cache:/opt/app-root/src/.cache/huggingface \
  ghcr.io/williamcaban/audio-transcriber:latest
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

## How it works

1. Pick a model (and, for Qwen3-ASR, optionally a source language) in the
   UI, then upload -> saved to `data/<job_id>/source.*`
2. `ffmpeg` extracts a 16kHz mono wav
3. The wav is read into a float32 array and split into fixed-size chunks
   (10 minutes for Parakeet, 30 seconds for Qwen3-ASR's native encoder
   window; no overlap — a word occasionally split at a boundary is an
   accepted tradeoff for keeping this simple); each chunk goes through that
   backend's `model.generate()`
4. Each backend turns its chunk output into the same `Segment(start, end,
   text)` shape (`app/asr/common.py`) regardless of how it got there --
   Parakeet's native per-token timestamps grouped into readable lines,
   Qwen3-ASR's one segment per 30s chunk -- and the raw segments are saved
   as `data/<job_id>/segments.json`
5. The browser preview (scrollable, fixed-height, so a long transcript
   doesn't stretch the page) and both download formats are all rendered from
   that same JSON on demand -- toggle "Timestamps" to switch between one
   timestamped line per segment and one continuous paragraph, "Copy text"
   copies the plain-text transcript to the clipboard, and you can download
   as Markdown (`.md`) or Word (`.docx`), in either timestamp mode, without
   re-running the model

Jobs are tracked in memory only (`app/jobs.py`) -- this is a personal
single-user tool, not a multi-tenant service. Restarting the server drops
any in-flight job.

### Model loading and availability

Only the default backend (Qwen3-ASR) is warmed up at startup, in a
background thread -- not blocking, so the server accepts connections (and
the UI loads) immediately rather than hanging until a multi-GB first-run
download finishes. `GET /api/models` reports each backend's state
(`not_loaded`/`loading`/`ready`/`error`) and, while loading, how many
seconds it's been at it; the UI polls this every 3s and disables uploading
with a visible banner for whichever model is currently selected and still
loading (or shows the error if loading failed). This is availability
status, not byte-level download progress -- huggingface_hub doesn't expose
file-level progress through a simple public callback, and a fake percentage
would be worse than an honest "loading, Ns so far."

The *other* backend (Parakeet, if Qwen3-ASR is selected) stays
`not_loaded` and loads lazily on its first actual use -- its first request
is just slower while that happens, folded into the job's existing
"Transcribing..." status rather than its own banner.

### Cancelling a running job

Transcription can take a while, especially for long files on CPU, so there's
a timer (elapsed time, ticking live) and a Cancel button while a job is in
progress. Cancellation is cooperative, not preemptive: `POST
/api/jobs/{id}/cancel` just sets a flag, and the running backend checks it
*between* chunks, not inside one. That means response time depends on chunk
size -- Qwen3-ASR (30s chunks) stops within moments; Parakeet (10-minute
chunks) can take up to the remainder of whatever chunk is in flight.
Whatever chunks finished before the flag was noticed are kept and become a
downloadable partial transcript, not discarded.

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

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pytest -v             # tests (loads the real models, no mocks)
```

`tests/test_parakeet.py` and `tests/test_qwen3.py` are end-to-end smoke
tests, not unit tests with the models mocked out: they load the real models
and transcribe a real speech sample (`tests/fixtures/libri_sample.wav`,
from the public LibriSpeech dummy dataset). That's deliberate -- every bug
this project has hit so far (missing `librosa`/`accelerate`, a token-joining
spacing bug that garbled every word, a mis-nested processor kwarg) only
showed up once a model actually loaded and ran; a mocked test would have
passed right through all of them.

CI (`.github/workflows/ci.yml`) runs both lint and the test suite on every
push to `main` and every PR. GitHub Actions are pinned to commit SHAs, not
mutable tags.
