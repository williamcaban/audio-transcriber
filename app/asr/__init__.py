"""ASR backend registry.

Each backend module exposes: MODEL_NAME, DISPLAY_NAME, SUPPORTS_LANGUAGE_SELECTION,
SUPPORTED_LANGUAGES (dict of code -> display name, empty if selection isn't
supported), load_model(), and transcribe(wav_path, duration_seconds, language) ->
(list[Segment], detected_language).
"""

from . import parakeet, qwen3
from .common import Segment, to_docx, to_markdown

DEFAULT_BACKEND = "qwen3"

# Dict order matters here, not just for iteration: the frontend populates the
# Model dropdown from /api/models in this order, and relies on the default
# being first so it's also the browser's default-selected <option>.
BACKENDS = {
    "qwen3": qwen3,
    "parakeet": parakeet,
}

__all__ = ["BACKENDS", "DEFAULT_BACKEND", "Segment", "to_docx", "to_markdown"]
