# How it works

1. Pick a model (and, for Qwen3-ASR, optionally a source language) in the
   UI, then upload -> saved to `data/<job_id>/source.*`
2. `ffmpeg` extracts a 16kHz mono wav
3. The wav is read into a float32 array and split into fixed-size chunks
   (10 minutes for Parakeet, 30 seconds for Qwen3-ASR's native encoder
   window; no overlap -- a word occasionally split at a boundary is an
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

## Model loading and availability

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

## Cancelling a running job

Transcription can take a while, especially for long files on CPU, so there's
a timer (elapsed time, ticking live) and a Cancel button while a job is in
progress. Cancellation is cooperative, not preemptive: `POST
/api/jobs/{id}/cancel` just sets a flag, and the running backend checks it
*between* chunks, not inside one. That means response time depends on chunk
size -- Qwen3-ASR (30s chunks) stops within moments; Parakeet (10-minute
chunks) can take up to the remainder of whatever chunk is in flight.
Whatever chunks finished before the flag was noticed are kept and become a
downloadable partial transcript, not discarded.
