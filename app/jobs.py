"""In-memory job store for transcription requests.

Single-process and non-persistent by design: this is a personal tool, not a
multi-user service. Restarting the server loses in-flight job state (the
uploaded file and any partial output are cleaned up too, which is fine).
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting_audio"
    TRANSCRIBING = "transcribing"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    original_filename: str
    status: JobStatus = JobStatus.QUEUED
    error: Optional[str] = None
    markdown_path: Optional[Path] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


_jobs: dict[str, Job] = {}
_lock = Lock()


def create_job(original_filename: str) -> Job:
    job = Job(id=str(uuid.uuid4()), original_filename=original_filename)
    with _lock:
        _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs[job_id]
        for key, value in fields.items():
            setattr(job, key, value)
