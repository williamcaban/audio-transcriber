# Audio Transcriber

Small FastAPI app that transcribes audio and video files to Markdown using
NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
model (NeMo toolkit).

## Language coverage

Parakeet TDT 0.6B v3 covers 25 European languages with automatic language
detection, including **Spanish**, **English**, and **French**. It does
**not** support Guaraní — that's a known gap for now, not a bug. If Guaraní
transcription becomes a real need, this would need a second model wired in
(e.g. Whisper large-v3, which has weak but nonzero Guaraní coverage).

## Requirements

- Python 3.10 or 3.11 (NeMo's dependency chain is not yet reliable on 3.12+)
- `ffmpeg` and `ffprobe` on PATH (audio/video extraction)
- A CUDA GPU is strongly recommended. It will run on CPU, but slowly, and
  long files may be impractical.

## Setup

```bash
python3.11 -m venv .venv
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
3. NeMo's `ASRModel.transcribe(..., timestamps=True)` runs inference;
   files longer than 10 minutes automatically switch the encoder to local
   attention (lower memory, slight quality tradeoff) per the model card's
   long-audio guidance
4. Segments are rendered into a timestamped Markdown transcript, served back
   to the browser and available as a `.md` download

Jobs are tracked in memory only (`app/jobs.py`) — this is a personal
single-user tool, not a multi-tenant service. Restarting the server drops
any in-flight job.

## Known limitations / next steps

- No persistent job history — add a DB if you want transcripts to survive restarts.
- Single model instance processes one job at a time (background tasks run
  sequentially against the one loaded model); fine for personal use, not for
  concurrent multi-user load.
- No forced language hint — the model auto-detects; if it ever picks the
  wrong language for a mixed-language file, a manual override param would be
  the fix.
