"""FastAPI app: upload audio/video, transcribe with a selectable ASR backend, return Markdown.

Flow: POST /api/jobs starts a background job (extract audio -> transcribe ->
persist segments as JSON), the client polls GET /api/jobs/{id} for status,
then fetches a rendered view or downloads a file -- both support toggling
timestamps on/off and (for downloads) choosing markdown or docx, rendered
on demand from the persisted segments rather than precomputed up front.

Two ASR backends are available (see app/asr/__init__.py): Parakeet TDT
(native word-level timestamps, no language control) and Qwen3-ASR (real
language forcing/detection, only chunk-level timestamps). Only the default
backend is warmed up at startup; the other loads lazily on first use.
"""

import json
import logging
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import jobs, media
from .asr import BACKENDS, DEFAULT_BACKEND, Segment, to_docx, to_markdown
from .asr import status as backend_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcriber.api")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
STATIC_DIR = REPO_ROOT / "static"
DATA_DIR.mkdir(exist_ok=True)

DOWNLOAD_MEDIA_TYPES = {
    "md": "text/markdown",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

app = FastAPI(title="Audio Transcriber")


def _ensure_loaded(key: str) -> None:
    """Load a backend's model, tracking status for /api/models to report.

    Safe to call repeatedly: each backend's own load_model() is a no-op once
    loaded (guarded by its own lock), so this just adds status bookkeeping
    around it.
    """
    backend_status.mark_loading(key)
    try:
        BACKENDS[key].load_model()
        backend_status.mark_ready(key)
    except Exception as exc:
        backend_status.mark_error(key, str(exc))
        raise


@app.on_event("startup")
def _warm_up_model() -> None:
    # Load only the default backend, and do it in a background thread rather
    # than blocking startup -- a blocking call here means uvicorn doesn't
    # start accepting connections until the (multi-GB, first-run) download
    # finishes, so the UI can't even load to show that it's happening. The
    # other backend loads lazily on its own first use.
    threading.Thread(
        target=_ensure_loaded, args=(DEFAULT_BACKEND,), daemon=True
    ).start()


@app.get("/api/models")
async def list_models():
    return {
        key: {
            "name": backend.DISPLAY_NAME,
            "supports_language_selection": backend.SUPPORTS_LANGUAGE_SELECTION,
            "languages": backend.SUPPORTED_LANGUAGES,
            **backend_status.get(key),
        }
        for key, backend in BACKENDS.items()
    }


def _process_job(job_id: str, upload_path: Path) -> None:
    job_dir = upload_path.parent
    try:
        job = jobs.get_job(job_id)
        backend = BACKENDS[job.backend]

        if job.cancel_event.is_set():
            jobs.update_job(
                job_id, status=jobs.JobStatus.CANCELLED, finished_at=datetime.now(UTC)
            )
            return

        jobs.update_job(job_id, status=jobs.JobStatus.EXTRACTING)
        wav_path = job_dir / "audio.wav"
        media.extract_audio(str(upload_path), str(wav_path))
        duration = media.probe_duration_seconds(str(wav_path))

        jobs.update_job(job_id, status=jobs.JobStatus.TRANSCRIBING)
        if backend_status.get(job.backend)["state"] != "ready":
            _ensure_loaded(job.backend)
        language = job.language if backend.SUPPORTS_LANGUAGE_SELECTION else None
        segments, detected_language = backend.transcribe(
            str(wav_path), duration, language, cancel_event=job.cancel_event
        )

        segments_path = job_dir / "segments.json"
        segments_path.write_text(
            json.dumps(
                {
                    "detected_language": detected_language,
                    "model_name": backend.MODEL_NAME,
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in segments
                    ],
                }
            ),
            encoding="utf-8",
        )

        # Cancellation is cooperative (checked between chunks), so a
        # cancelled job can still have finished some chunks -- write them
        # out as a partial transcript rather than discarding the work.
        final_status = (
            jobs.JobStatus.CANCELLED
            if job.cancel_event.is_set()
            else jobs.JobStatus.DONE
        )
        jobs.update_job(
            job_id,
            status=final_status,
            segments_path=segments_path,
            finished_at=datetime.now(UTC),
        )
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        jobs.update_job(
            job_id,
            status=jobs.JobStatus.ERROR,
            error=str(exc),
            finished_at=datetime.now(UTC),
        )
    finally:
        upload_path.unlink(missing_ok=True)
        wav_candidate = job_dir / "audio.wav"
        wav_candidate.unlink(missing_ok=True)


_RESULT_READY_STATUSES = {jobs.JobStatus.DONE, jobs.JobStatus.CANCELLED}


def _require_result(job_id: str) -> jobs.Job:
    job = jobs.get_job(job_id)
    if (
        job is None
        or job.status not in _RESULT_READY_STATUSES
        or job.segments_path is None
    ):
        raise HTTPException(404, "transcript not ready")
    return job


def _load_result(job: jobs.Job) -> tuple[list[Segment], str | None, str]:
    data = json.loads(job.segments_path.read_text(encoding="utf-8"))
    segments = [Segment(**s) for s in data["segments"]]
    return segments, data["detected_language"], data["model_name"]


@app.post("/api/jobs")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    backend: str = Form(DEFAULT_BACKEND),
    language: str = Form("auto"),
):
    if not file.filename:
        raise HTTPException(400, "missing filename")
    if backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend: {backend!r}")

    backend_module = BACKENDS[backend]
    resolved_language = None
    if backend_module.SUPPORTS_LANGUAGE_SELECTION and language != "auto":
        if language not in backend_module.SUPPORTED_LANGUAGES:
            raise HTTPException(
                400, f"unsupported language code for {backend!r}: {language!r}"
            )
        resolved_language = language

    job = jobs.create_job(file.filename, backend=backend, language=resolved_language)
    job_dir = DATA_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    upload_path = job_dir / f"source{Path(file.filename).suffix}"
    with upload_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    background_tasks.add_task(_process_job, job.id, upload_path)
    return {"job_id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    elapsed = (job.finished_at or datetime.now(UTC)) - job.created_at
    return {
        "job_id": job.id,
        "filename": job.original_filename,
        "status": job.status,
        "error": job.error,
        "elapsed_seconds": elapsed.total_seconds(),
    }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status in (
        jobs.JobStatus.QUEUED,
        jobs.JobStatus.EXTRACTING,
        jobs.JobStatus.TRANSCRIBING,
    ):
        job.cancel_event.set()
    return {"job_id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}/markdown")
async def job_markdown(job_id: str, timestamps: bool = True):
    job = _require_result(job_id)
    segments, detected_language, model_name = _load_result(job)
    markdown = to_markdown(
        job.original_filename, segments, detected_language, model_name, timestamps
    )
    return JSONResponse({"markdown": markdown})


@app.get("/api/jobs/{job_id}/download")
async def job_download(
    job_id: str, format: Literal["md", "docx"] = "md", timestamps: bool = True
):
    job = _require_result(job_id)
    segments, detected_language, model_name = _load_result(job)
    stem = Path(job.original_filename).stem

    if format == "md":
        content = to_markdown(
            job.original_filename, segments, detected_language, model_name, timestamps
        ).encode("utf-8")
    else:
        content = to_docx(
            job.original_filename, segments, detected_language, model_name, timestamps
        )

    return Response(
        content=content,
        media_type=DOWNLOAD_MEDIA_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="{stem}.{format}"'},
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
