"""FastAPI app: upload audio/video, transcribe with Parakeet TDT, return Markdown.

Flow: POST /api/jobs starts a background job (extract audio -> transcribe ->
write transcript.md), the client polls GET /api/jobs/{id} for status, then
fetches the rendered markdown or downloads the .md file.
"""

import logging
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import jobs, media, transcriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcriber.api")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
STATIC_DIR = REPO_ROOT / "static"
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Audio Transcriber")


@app.on_event("startup")
def _warm_up_model() -> None:
    # Load at startup, in the main thread, so the first upload isn't the one
    # that pays the (large) model download/load cost.
    transcriber.load_model()


def _process_job(job_id: str, upload_path: Path) -> None:
    job_dir = upload_path.parent
    try:
        job = jobs.get_job(job_id)

        jobs.update_job(job_id, status=jobs.JobStatus.EXTRACTING)
        wav_path = job_dir / "audio.wav"
        media.extract_audio(str(upload_path), str(wav_path))
        duration = media.probe_duration_seconds(str(wav_path))

        jobs.update_job(job_id, status=jobs.JobStatus.TRANSCRIBING)
        segments, detected_language = transcriber.transcribe(str(wav_path), duration)

        markdown = transcriber.to_markdown(
            job.original_filename, segments, detected_language
        )
        markdown_path = job_dir / "transcript.md"
        markdown_path.write_text(markdown, encoding="utf-8")

        jobs.update_job(job_id, status=jobs.JobStatus.DONE, markdown_path=markdown_path)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        jobs.update_job(job_id, status=jobs.JobStatus.ERROR, error=str(exc))
    finally:
        upload_path.unlink(missing_ok=True)
        wav_candidate = job_dir / "audio.wav"
        wav_candidate.unlink(missing_ok=True)


@app.post("/api/jobs")
async def submit_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "missing filename")

    job = jobs.create_job(file.filename)
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
    return {
        "job_id": job.id,
        "filename": job.original_filename,
        "status": job.status,
        "error": job.error,
    }


@app.get("/api/jobs/{job_id}/markdown")
async def job_markdown(job_id: str):
    job = jobs.get_job(job_id)
    if job is None or job.status != jobs.JobStatus.DONE or job.markdown_path is None:
        raise HTTPException(404, "transcript not ready")
    return JSONResponse({"markdown": job.markdown_path.read_text(encoding="utf-8")})


@app.get("/api/jobs/{job_id}/download")
async def job_download(job_id: str):
    job = jobs.get_job(job_id)
    if job is None or job.status != jobs.JobStatus.DONE or job.markdown_path is None:
        raise HTTPException(404, "transcript not ready")
    name = Path(job.original_filename).stem + ".md"
    return FileResponse(job.markdown_path, filename=name, media_type="text/markdown")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
