# Audio Transcriber

Small FastAPI app that transcribes audio and video files to Markdown using
NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
model, via its `transformers`-native port (`AutoModelForTDT`).

## Language coverage

Parakeet TDT 0.6B v3 covers 25 European languages with automatic language
detection, including **Spanish**, **English**, and **French**. It does
**not** support Guaraní — that's a known gap for now, not a bug. If Guaraní
transcription becomes a real need, this would need a second model wired in
(e.g. Whisper large-v3, which has weak but nonzero Guaraní coverage).

## Requirements

- Python 3.13+
- `ffmpeg` and `ffprobe` on PATH (audio/video extraction)
- A CUDA GPU is strongly recommended. It will run on CPU, but slowly, and
  long files may be impractical.

Uses the `transformers`-native path (`AutoModelForTDT`/`AutoProcessor`)
rather than the NeMo toolkit, specifically because NeMo's dependency chain
(pytorch-lightning, lhotse, megatron-core, ...) lags new Python releases by
months and this project targets 3.13+. The tradeoff: Parakeet TDT support in
`transformers` is very recent and not yet in a stable release, so
`requirements.txt` installs it straight from the GitHub `main` branch — pin
it to a commit SHA (see the comment in that file) once you've confirmed a
version that works for you.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate

# Install a torch build matching your CUDA version first if the default
# wheel isn't right for your GPU: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 (or `http://<host>:8000` if running on a
remote GPU box) and drop in a file. The model downloads from Hugging Face on
first startup (a few GB) and is cached by `huggingface_hub` after that.

## How it works

1. Upload -> saved to `data/<job_id>/source.*`
2. `ffmpeg` extracts a 16kHz mono wav
3. The wav is read into a float32 array and split into fixed 10-minute
   chunks (no overlap — a word occasionally split at a boundary is an
   accepted tradeoff for keeping this simple); each chunk goes through
   `model.generate()` and `processor.decode(..., durations=...)`
4. Per-token/word timestamps are grouped into readable lines (new line every
   ~15s or at sentence-ending punctuation) and rendered into a timestamped
   Markdown transcript, served back to the browser and available as a `.md`
   download

Jobs are tracked in memory only (`app/jobs.py`) — this is a personal
single-user tool, not a multi-tenant service. Restarting the server drops
any in-flight job.

## Known limitations / next steps

- The transformers port of Parakeet TDT is pre-release; the exact shape of
  the per-token timestamp output isn't pinned down by stable docs, so
  `transcriber._entries_to_segments` is deliberately defensive and falls
  back to one untimed block of text per chunk if parsing fails. If that
  fallback is firing in practice, re-check the model card for the current
  output format and tighten the parser.
- No persistent job history — add a DB if you want transcripts to survive restarts.
- Single model instance processes one job at a time (background tasks run
  sequentially against the one loaded model); fine for personal use, not for
  concurrent multi-user load.
- No forced language hint — the model auto-detects; if it ever picks the
  wrong language for a mixed-language file, a manual override param would be
  the fix.
- Long files are chunked at fixed 10-minute boundaries with no overlap, so a
  word spanning a boundary can be slightly garbled. Not addressed in v1.
