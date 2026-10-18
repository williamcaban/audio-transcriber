"""End-to-end smoke test: loads the real model and transcribes a real sample.

Deliberately not mocked. The bugs this project has actually hit so far
(missing `librosa`/`accelerate`, a token-joining spacing bug that garbled
every word) only show up once `AutoProcessor`/`AutoModelForTDT` are loaded
for real and run against real audio -- a test that mocks the model would
have passed right through all of them.
"""

from pathlib import Path

from app.asr.parakeet import transcribe

FIXTURE = Path(__file__).parent / "fixtures" / "libri_sample.wav"

# LibriSpeech dummy dataset reference text for this sample (uppercase in the
# source dataset; compared case-insensitively below since exact casing isn't
# the point of this test).
REFERENCE_WORDS = ["quilter", "apostle", "middle", "classes", "welcome", "gospel"]


def test_transcribe_matches_reference_words():
    segments, _detected_language = transcribe(str(FIXTURE), duration_seconds=5.855)

    assert segments, "expected at least one segment from a 5.8s speech sample"

    full_text = " ".join(segment.text for segment in segments).lower()

    missing = [word for word in REFERENCE_WORDS if word not in full_text]
    assert not missing, (
        f"transcript missing expected words {missing}: got {full_text!r}"
    )

    # Regression guard for the token-joining spacing bug: real words should
    # appear as whole words, not broken into single-letter fragments.
    assert "m ister" not in full_text
    assert "q u" not in full_text
