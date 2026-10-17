"""Parakeet TDT 0.6B v3 wrapper: load once, transcribe a wav into timestamped segments.

Uses the NeMo toolkit (not the transformers path) because NeMo exposes
segment-level timestamps and a documented long-audio mode out of the box,
both of which the Markdown output below depends on.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import torch

logger = logging.getLogger("transcriber.model")

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"

# Model card: up to ~24 min with full attention on an A100-80GB. Smaller GPUs
# need local attention sooner, so default conservatively; override via env
# if you know your GPU handles longer full-attention passes.
LONG_AUDIO_SECONDS = 600
LOCAL_ATTENTION_CONTEXT = (256, 256)

_model = None
_model_lock = threading.Lock()
_local_attention_active = False
_device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Segment:
    start: float
    end: float
    text: str


def load_model():
    global _model
    with _model_lock:
        if _model is None:
            import nemo.collections.asr as nemo_asr

            logger.info("Loading %s on %s ...", MODEL_NAME, _device)
            if _device == "cpu":
                logger.warning(
                    "No CUDA GPU detected. Parakeet will run on CPU: expect it to be "
                    "much slower than real-time, and long files may be impractical."
                )
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
            model = model.to(_device)
            model.eval()
            _model = model
            logger.info("Model loaded.")
    return _model


def _enable_long_audio_mode():
    global _local_attention_active
    if not _local_attention_active:
        logger.info("Long audio detected: switching encoder to local attention")
        _model.encoder.set_attention_model("rel_pos_local_attn", LOCAL_ATTENTION_CONTEXT)
        _local_attention_active = True


def transcribe(wav_path: str, duration_seconds: float) -> tuple[list[Segment], Optional[str]]:
    model = load_model()

    if duration_seconds > LONG_AUDIO_SECONDS:
        _enable_long_audio_mode()

    with torch.inference_mode():
        output = model.transcribe([wav_path], timestamps=True)

    hypothesis = output[0]
    detected_language = getattr(hypothesis, "lang", None)

    segments: list[Segment] = []
    timestamp_data = getattr(hypothesis, "timestamp", None) or {}
    for item in timestamp_data.get("segment", []):
        segments.append(Segment(start=item["start"], end=item["end"], text=item["segment"].strip()))

    if not segments:
        # Fallback for model/version combos that don't return segment stamps.
        segments.append(Segment(start=0.0, end=duration_seconds, text=hypothesis.text.strip()))

    return segments, detected_language


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_markdown(original_filename: str, segments: list[Segment], detected_language: Optional[str]) -> str:
    lines = [f"# Transcript: {original_filename}", ""]
    lines.append(f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- **Model:** {MODEL_NAME}")
    if detected_language:
        lines.append(f"- **Detected language:** {detected_language}")
    lines.append("")

    for seg in segments:
        lines.append(f"**[{_format_timestamp(seg.start)}]** {seg.text}")
        lines.append("")

    return "\n".join(lines)
