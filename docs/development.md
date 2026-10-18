# Development

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pytest -v             # tests (loads the real models, no mocks)
```

`tests/test_parakeet.py` and `tests/test_qwen3.py` are end-to-end smoke
tests, not unit tests with the models mocked out: they load the real models
and transcribe a real speech sample (`tests/fixtures/libri_sample.wav`,
from the public LibriSpeech dummy dataset). That's deliberate -- every bug
this project has hit so far (missing `librosa`/`accelerate`, a token-joining
spacing bug that garbled every word, a mis-nested processor kwarg) only
showed up once a model actually loaded and ran; a mocked test would have
passed right through all of them.

CI (`.github/workflows/ci.yml`) runs `lint` and `test` on every push to
`main` and every PR, for fast iteration feedback. GitHub Actions are pinned
to commit SHAs, not mutable tags.

Building and publishing the actual deployable artifacts (container image,
macOS `.app`) is a separate workflow, `.github/workflows/release.yml` -- see
[Releases and dev builds](../README.md#releases-and-dev-builds) in the main
README -- so those heavier builds (the macOS one in particular has taken
several minutes to over half an hour) don't slow down every PR.
