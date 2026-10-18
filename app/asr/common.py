"""Shared pieces across ASR backends: the Segment type, audio I/O helpers,
device detection, and Markdown/Word rendering.

Rendering is backend-agnostic by design: every backend's transcribe()
returns the same `list[Segment]` + detected-language shape, so a single
to_markdown()/to_docx() serves both Parakeet's word-level timestamps and
Qwen3-ASR's coarser chunk-level ones without caring which produced them.
"""

import io
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import soundfile as sf
import torch


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        # Apple Silicon. These models' transformers ports are very new, so
        # MPS op coverage isn't guaranteed -- if generate() errors on an
        # unsupported op, force CPU with PYTORCH_ENABLE_MPS_FALLBACK=1 or by
        # overriding this function's return value.
        return "mps"
    return "cpu"


@dataclass
class Segment:
    start: float
    end: float
    text: str


def read_mono_audio(wav_path: str, target_sample_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        # media.extract_audio already resamples to 16kHz; this only fires if
        # that contract is ever violated, so fail loudly instead of silently
        # mis-transcribing.
        raise ValueError(f"expected {target_sample_rate}Hz audio, got {sample_rate}Hz")
    return audio


def chunk_audio(audio: np.ndarray, sample_rate: int, chunk_seconds: int):
    chunk_len = chunk_seconds * sample_rate
    for start_sample in range(0, len(audio), chunk_len):
        start_time = start_sample / sample_rate
        yield start_time, audio[start_sample : start_sample + chunk_len]


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _metadata_fields(
    model_name: str, detected_language: str | None
) -> list[tuple[str, str]]:
    fields = [
        ("Generated", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")),
        ("Model", model_name),
    ]
    if detected_language:
        fields.append(("Detected language", detected_language))
    return fields


def to_markdown(
    original_filename: str,
    segments: list[Segment],
    detected_language: str | None,
    model_name: str,
    with_timestamps: bool = True,
) -> str:
    lines = [f"# Transcript: {original_filename}", ""]
    lines += [
        f"- **{label}:** {value}"
        for label, value in _metadata_fields(model_name, detected_language)
    ]
    lines.append("")

    if with_timestamps:
        for seg in segments:
            lines.append(f"**[{_format_timestamp(seg.start)}]** {seg.text}")
            lines.append("")
    else:
        lines.append(" ".join(seg.text for seg in segments))

    return "\n".join(lines)


def to_docx(
    original_filename: str,
    segments: list[Segment],
    detected_language: str | None,
    model_name: str,
    with_timestamps: bool = True,
) -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading(f"Transcript: {original_filename}", level=1)
    meta_paragraph = doc.add_paragraph()
    meta_text = " | ".join(
        f"{label}: {value}"
        for label, value in _metadata_fields(model_name, detected_language)
    )
    meta_paragraph.add_run(meta_text).italic = True

    if with_timestamps:
        for seg in segments:
            p = doc.add_paragraph()
            p.add_run(f"[{_format_timestamp(seg.start)}] ").bold = True
            p.add_run(seg.text)
    else:
        doc.add_paragraph(" ".join(seg.text for seg in segments))

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
