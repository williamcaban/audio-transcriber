"""Parakeet TDT 0.6B v3 wrapper: load once, transcribe a wav into timestamped segments.

Uses the `transformers`-native port (`AutoModelForTDT`/`AutoProcessor`)
rather than the NeMo toolkit. NeMo's own dependency chain (pytorch-lightning,
lhotse, megatron-core, ...) lags new Python releases by months; transformers
tracks them fast, which matters here since the project targets Python 3.13+.

Tradeoff: Parakeet TDT support landed in transformers very recently and
isn't in a stable release yet (see README for the `pip install` from
source), and the long-audio "local attention" trick NeMo exposes isn't part
of this API. Long files are handled here instead by chunking the raw audio
ourselves before inference (see `CHUNK_SECONDS`).
"""

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("transcriber.model")

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"

# Conservative default so it behaves on modest GPUs too; override if you know
# your hardware handles longer passes. No overlap between chunks: a word
# occasionally split at a chunk boundary is an accepted tradeoff for keeping
# this simple.
CHUNK_SECONDS = 600


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        # Apple Silicon. This model's transformers port is very new, so MPS
        # op coverage isn't guaranteed -- if generate() errors on an
        # unsupported op, force CPU with PYTORCH_ENABLE_MPS_FALLBACK=1 or by
        # overriding this function's return value.
        return "mps"
    return "cpu"


_model = None
_processor = None
_model_lock = threading.Lock()
_device = _detect_device()


@dataclass
class Segment:
    start: float
    end: float
    text: str


def load_model():
    global _model, _processor
    with _model_lock:
        if _model is None:
            from transformers import AutoModelForTDT, AutoProcessor

            logger.info("Loading %s on %s ...", MODEL_NAME, _device)
            if _device == "cpu":
                logger.warning(
                    "No CUDA GPU detected. Parakeet will run on CPU: expect it to be "
                    "much slower than real-time, and long files may be impractical."
                )
            _processor = AutoProcessor.from_pretrained(MODEL_NAME)
            _model = AutoModelForTDT.from_pretrained(
                MODEL_NAME, dtype="auto", device_map=_device
            )
            _model.eval()
            logger.info("Model loaded.")
    return _model, _processor


def _read_mono_audio(wav_path: str, target_sample_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        # media.extract_audio already resamples to 16kHz; this only fires if
        # that contract is ever violated, so fail loudly instead of silently
        # mis-transcribing.
        raise ValueError(f"expected {target_sample_rate}Hz audio, got {sample_rate}Hz")
    return audio


def _chunk_audio(audio: np.ndarray, sample_rate: int, chunk_seconds: int):
    chunk_len = chunk_seconds * sample_rate
    for start_sample in range(0, len(audio), chunk_len):
        start_time = start_sample / sample_rate
        yield start_time, audio[start_sample : start_sample + chunk_len]


def _transcribe_chunk(
    model, processor, audio_chunk: np.ndarray, sample_rate: int
) -> tuple[str, list]:
    inputs = processor(audio=[audio_chunk], sampling_rate=sample_rate)
    inputs.to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        output = model.generate(**inputs, return_dict_in_generate=True)

    text, timestamps = processor.decode(
        output.sequences,
        durations=output.durations,
        skip_special_tokens=True,
    )
    text = text[0] if isinstance(text, list) else text
    entries = (
        timestamps[0]
        if timestamps and isinstance(timestamps, list)
        else (timestamps or [])
    )
    return text, entries


def _entries_to_segments(
    entries: list, chunk_start: float, chunk_text: str
) -> list[Segment]:
    """Group per-token/word timestamp entries into readable lines.

    The exact shape of `entries` isn't pinned down by stable docs yet (this
    API is pre-release) so this is deliberately defensive: any entry that
    doesn't look like {start, end/duration, text} just gets skipped, and if
    nothing usable comes out of it at all, the caller falls back to one
    block of plain text for the chunk.
    """
    segments: list[Segment] = []
    words: list[str] = []
    seg_start: float | None = None
    last_end = chunk_start

    def flush():
        if words:
            # Tokens already carry their own leading space where a new word
            # starts (e.g. "m", "ister", " Qu", "il", "ter" -> "mister Quilter"),
            # so joining them with an extra separator would break words apart.
            segments.append(
                Segment(start=seg_start, end=last_end, text="".join(words).strip())
            )
            words.clear()

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        token = entry.get("token") or entry.get("word") or entry.get("text")
        start = entry.get("start", entry.get("start_time"))
        end = entry.get("end", entry.get("end_time"))
        if token is None or start is None:
            continue
        start = chunk_start + float(start)
        end = chunk_start + float(end) if end is not None else start

        if seg_start is None:
            seg_start = start
        words.append(str(token))
        last_end = end

        is_long_enough = (last_end - seg_start) >= 15
        ends_sentence = str(token).strip().endswith((".", "?", "!"))
        if is_long_enough or ends_sentence:
            flush()
            seg_start = None

    flush()
    return segments


def transcribe(
    wav_path: str, duration_seconds: float
) -> tuple[list[Segment], str | None]:
    model, processor = load_model()
    sample_rate = processor.feature_extractor.sampling_rate

    audio = _read_mono_audio(wav_path, sample_rate)

    all_segments: list[Segment] = []
    for chunk_start, audio_chunk in _chunk_audio(audio, sample_rate, CHUNK_SECONDS):
        if len(audio_chunk) == 0:
            continue
        text, entries = _transcribe_chunk(model, processor, audio_chunk, sample_rate)

        segments = []
        try:
            segments = _entries_to_segments(entries, chunk_start, text)
        except Exception:
            logger.warning(
                "Could not parse per-token timestamps for this chunk; using plain text",
                exc_info=True,
            )

        if not segments and text.strip():
            chunk_end = chunk_start + len(audio_chunk) / sample_rate
            segments = [Segment(start=chunk_start, end=chunk_end, text=text.strip())]

        all_segments.extend(segments)

    # This transformers-native path doesn't expose a detected-language field
    # the way the NeMo API did; leave room for it without inventing a value.
    detected_language = None

    return all_segments, detected_language


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_markdown(
    original_filename: str, segments: list[Segment], detected_language: str | None
) -> str:
    lines = [f"# Transcript: {original_filename}", ""]
    lines.append(f"- **Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- **Model:** {MODEL_NAME}")
    if detected_language:
        lines.append(f"- **Detected language:** {detected_language}")
    lines.append("")

    for seg in segments:
        lines.append(f"**[{_format_timestamp(seg.start)}]** {seg.text}")
        lines.append("")

    return "\n".join(lines)
