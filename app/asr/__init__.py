"""ASR backend registry.

Each backend module exposes: MODEL_NAME, DISPLAY_NAME, SUPPORTS_LANGUAGE_SELECTION,
SUPPORTED_LANGUAGES (dict of code -> display name, empty if selection isn't
supported), load_model(), and transcribe(wav_path, duration_seconds, language) ->
(list[Segment], detected_language).
"""

from . import parakeet, qwen3
from .common import Segment, to_docx, to_markdown

DEFAULT_BACKEND = "parakeet"

BACKENDS = {
    "parakeet": parakeet,
    "qwen3": qwen3,
}

__all__ = ["BACKENDS", "DEFAULT_BACKEND", "Segment", "to_docx", "to_markdown"]
