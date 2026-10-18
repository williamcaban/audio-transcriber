"""/api/version is a thin read of app.__version__ -- fast, no model load.

Calls the route function directly rather than going through TestClient,
since TestClient's lifespan context would trigger the real startup event
(which kicks off a background model-loading thread) for no reason here.
"""

import asyncio

from app import __version__
from app.main import get_version


def test_api_version_matches_the_single_source_of_truth():
    result = asyncio.run(get_version())
    assert result == {"name": "Audio Transcriber", "version": __version__}
