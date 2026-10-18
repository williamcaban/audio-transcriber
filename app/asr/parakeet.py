"""Parakeet TDT 0.6B v3 backend: native word-level timestamps, no language control.

Uses the `transformers`-native port (`AutoModelForTDT`/`AutoProcessor`)
rather than the NeMo toolkit. NeMo's own dependency chain (pytorch-lightning,
lhotse, megatron-core, ...) lags new Python releases by months; transformers
tracks them fast, which matters here since the project targets Python 3.13+.

Tradeoff: Parakeet TDT support landed in transformers very recently and
isn't in a stable release yet (see README for the `pip install` from
source), and the long-audio "local attention" trick NeMo exposes isn't part
of this API. Long files are handled here instead by chunking the raw audio
ourselves before inference (see `CHUNK_SECONDS`).

No language parameter: Parakeet has no API to force or detect a source
language (confirmed by reading processing_parakeet.py/generation_parakeet.py
in the installed package -- no language token, prompt mechanism, or output
field anywhere). `transcribe()` accepts `language` for interface parity with
other backends but ignores it.
"""

import logging
import threading
import warnings

import torch

from .common import Segment, chunk_audio, detect_device, read_mono_audio

logger = logging.getLogger("transcriber.parakeet")

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
DISPLAY_NAME = "NVIDIA Parakeet TDT 0.6B v3"
SUPPORTS_LANGUAGE_SELECTION = False
SUPPORTED_LANGUAGES: dict[str, str] = {}

# Conservative default so it behaves on modest GPUs too; override if you know
# your hardware handles longer passes. No overlap between chunks: a word
# occasionally split at a chunk boundary is an accepted tradeoff for keeping
# this simple.
CHUNK_SECONDS = 600

# Misleading, not informative: transformers' generic generate() code warns
# "using the model-agnostic default max_length" whenever max_new_tokens isn't
# passed explicitly, but Parakeet's own generate() override (see
# generation_parakeet.py in the installed package) always recomputes
# max_length from the real encoder sequence length *before* that warning
# fires -- so long audio already gets a correspondingly large token budget.
# Confirmed by the value scaling correctly with input duration in practice.
warnings.filterwarnings(
    "ignore",
    message=r"Using the model-agnostic default `max_length`",
    category=UserWarning,
)

_model = None
_processor = None
_model_lock = threading.Lock()
_device = detect_device()


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


def _transcribe_chunk(
    model, processor, audio_chunk, sample_rate: int
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
    wav_path: str,
    duration_seconds: float,
    language: str | None = None,
    cancel_event=None,
) -> tuple[list[Segment], str | None]:
    model, processor = load_model()
    sample_rate = processor.feature_extractor.sampling_rate

    audio = read_mono_audio(wav_path, sample_rate)

    all_segments: list[Segment] = []
    for chunk_start, audio_chunk in chunk_audio(audio, sample_rate, CHUNK_SECONDS):
        if cancel_event is not None and cancel_event.is_set():
            break
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
