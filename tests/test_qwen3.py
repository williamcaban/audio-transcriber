"""End-to-end smoke test for the Qwen3-ASR backend: real model, real audio.

Same philosophy as test_parakeet.py: no mocks. This already caught a real
bug (processor.apply_transcription_request warning about mis-nested kwargs)
that reading the source alone didn't surface.
"""

from pathlib import Path

from app.asr.qwen3 import transcribe

FIXTURE = Path(__file__).parent / "fixtures" / "libri_sample.wav"
REFERENCE_WORDS = ["quilter", "apostle", "middle", "classes", "welcome", "gospel"]


def test_transcribe_auto_detects_english():
    segments, detected_language = transcribe(str(FIXTURE), duration_seconds=5.855)

    assert segments, "expected at least one segment from a 5.8s speech sample"
    assert detected_language == "English"

    full_text = " ".join(segment.text for segment in segments).lower()
    missing = [word for word in REFERENCE_WORDS if word not in full_text]
    assert not missing, (
        f"transcript missing expected words {missing}: got {full_text!r}"
    )


def test_transcribe_with_forced_language_reports_that_language():
    segments, detected_language = transcribe(
        str(FIXTURE), duration_seconds=5.855, language="es"
    )

    assert segments
    # Forcing a language just conditions generation on it; it doesn't
    # magically translate the audio's actual (English) content, so this
    # only checks the reported language reflects what was forced.
    assert detected_language == "es"
