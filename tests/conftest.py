"""Shared fixtures for VideoSearch tests.

The canonical test video is Rick Astley - Never Gonna Give You Up.
It's a good test case: continuous music (validates silence detection fallback),
distinctive lyrics (validates transcript search), and known structure.
"""

import os
import subprocess
import tempfile

import pytest

# The video that started it all
RICK_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
RICK_TITLE = "Rick Astley - Never Gonna Give You Up"
RICK_DURATION_APPROX = 212  # ~3:32


@pytest.fixture(scope="session")
def rick_video(tmp_path_factory):
    """Download Rick Astley once per test session. Returns path to mp4.

    This is a session-scoped fixture so we only download once even if
    multiple tests use it. The file lives in a temp dir that gets
    cleaned up after the test session.
    """
    tmp_dir = str(tmp_path_factory.mktemp("rick"))
    output = os.path.join(tmp_dir, "rick.mp4")

    result = subprocess.run(
        [
            "yt-dlp",
            "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output,
            RICK_URL,
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"Could not download test video: {result.stderr[:200]}")

    return output


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary database path for test isolation."""
    return str(tmp_path / "test.db")


@pytest.fixture
def store(tmp_db):
    """Return a VideoStore with a temporary database."""
    from videosearch.store import VideoStore
    s = VideoStore(db_path=tmp_db)
    yield s
    s.close()
