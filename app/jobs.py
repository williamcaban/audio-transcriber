"""In-memory job store for transcription requests.

Single-process and non-persistent by design: this is a personal tool, not a
multi-user service. Restarting the server loses in-flight job state (the
uploaded file and any partial output are cleaned up too, which is fine).
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import Lock


class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting_audio"
    TRANSCRIBING = "transcribing"
    DONE = "done"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class Job:
    id: str
    original_filename: str
    backend: str
    language: str | None = None
    status: JobStatus = JobStatus.QUEUED
    error: str | None = None
    # Raw transcription result (segments + detected_language), persisted as
    # JSON so any output view/format can be rendered on demand without
    # re-running the model -- see asr.to_markdown/to_docx. Also written for
    # a cancelled job if any chunks finished before the cancel took effect,
    # so a partial transcript is still downloadable.
    segments_path: Path | None = None
    # Checked between chunks by the running backend; set by POST
    # /api/jobs/{id}/cancel. Cooperative, not preemptive: cancelling mid-chunk
    # still waits for that one chunk's generate() call to finish.
    cancel_event: threading.Event = field(default_factory=threading.Event)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None


_jobs: dict[str, Job] = {}
_lock = Lock()


def create_job(
    original_filename: str, backend: str, language: str | None = None
) -> Job:
    job = Job(
        id=str(uuid.uuid4()),
        original_filename=original_filename,
        backend=backend,
        language=language,
    )
    with _lock:
        _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs[job_id]
        for key, value in fields.items():
            setattr(job, key, value)
