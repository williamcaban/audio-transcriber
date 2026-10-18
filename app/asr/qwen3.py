"""Qwen3-ASR 1.7B backend: real language forcing/detection, no native timestamps.

Unlike Parakeet, Qwen3-ASR genuinely supports forcing a source language and
reports a detected language when none is forced (confirmed by reading
processing_qwen3_asr.py in the installed package: the language is forced by
prefilling the assistant turn with "language <NAME><asr_text>"; in
auto-detect mode the model generates that prefix itself and we parse it back
out). It ships in a stable transformers release, unlike Parakeet TDT.

The real tradeoff: this is an LLM-style model (Qwen3-Omni lineage) that just
emits transcription text -- there's no native per-token timestamp output the
way Parakeet's transducer architecture gives for free. Real word-level
timestamps would need a *second* model (Qwen3-ForcedAligner-0.6B, only 11 of
these 30 languages) and a second inference pass -- not implemented here.
Instead, each fixed-size chunk becomes one timestamped Segment, so the
timestamp toggle still does something useful (chunk-level granularity, not
word-level), without taking on that extra model/pipeline.

Its audio encoder also works in fixed windows (`feature_extractor.chunk_length`,
30s by default, Whisper-style) rather than Parakeet's ~10-minute chunks, so
long files mean substantially more chunks -- and therefore more generate()
calls -- than the Parakeet path.
"""

import logging
import threading

import torch

from .common import Segment, chunk_audio, detect_device, read_mono_audio

logger = logging.getLogger("transcriber.qwen3")

MODEL_NAME = "Qwen/Qwen3-ASR-1.7B-hf"
DISPLAY_NAME = "Qwen3-ASR 1.7B"
SUPPORTS_LANGUAGE_SELECTION = True

# The exact set this checkpoint was trained on (processing_qwen3_asr.py's
# LANGUAGE_CODE_TO_NAME) -- kept as our own copy rather than imported so a
# transformers upgrade that changes the installed mapping can't silently
# change what this app accepts without being noticed in review.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}

CHUNK_SECONDS = 30
MAX_NEW_TOKENS = 256

_model = None
_processor = None
_model_lock = threading.Lock()
_device = detect_device()


def load_model():
    global _model, _processor
    with _model_lock:
        if _model is None:
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            logger.info("Loading %s on %s ...", MODEL_NAME, _device)
            if _device == "cpu":
                logger.warning(
                    "No CUDA GPU detected. Qwen3-ASR will run on CPU: expect it to be "
                    "much slower than real-time, and long files may be impractical."
                )
            _processor = AutoProcessor.from_pretrained(MODEL_NAME)
            _model = AutoModelForMultimodalLM.from_pretrained(
                MODEL_NAME, dtype="auto", device_map=_device
            )
            _model.eval()
            logger.info("Model loaded.")
    return _model, _processor


def _transcribe_chunk(
    model, processor, audio_chunk, sample_rate: int, language: str | None
) -> tuple[str, str | None]:
    inputs = processor.apply_transcription_request(
        audio=audio_chunk,
        language=language,
        processor_kwargs={
            "audio_kwargs": {"sampling_rate": sample_rate},
            "common_kwargs": {"return_tensors": "pt"},
        },
    )
    inputs = inputs.to(model.device, dtype=model.dtype)
    prompt_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids = output_ids[:, prompt_length:]

    if language is not None:
        # Forced mode: the prompt already contains "language <NAME><asr_text>",
        # so the generated continuation is pure transcription text already.
        text = processor.decode(generated_ids[0], skip_special_tokens=True)
        return text.strip(), language
    # Auto-detect mode: nothing was prefilled, so the model generates the
    # full "language <NAME><asr_text>transcription" string itself.
    parsed = processor.decode(generated_ids[0], return_format="parsed")
    return parsed["transcription"].strip(), parsed["language"]


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
    detected_language: str | None = None

    for chunk_start, audio_chunk in chunk_audio(audio, sample_rate, CHUNK_SECONDS):
        if cancel_event is not None and cancel_event.is_set():
            break
        if len(audio_chunk) == 0:
            continue
        text, chunk_language = _transcribe_chunk(
            model, processor, audio_chunk, sample_rate, language
        )

        if detected_language is None:
            detected_language = chunk_language

        if text:
            chunk_end = chunk_start + len(audio_chunk) / sample_rate
            all_segments.append(Segment(start=chunk_start, end=chunk_end, text=text))

    return all_segments, detected_language
