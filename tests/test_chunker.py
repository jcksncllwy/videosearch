"""Tests for video chunking logic.

Unit tests use synthetic data. Integration tests use Rick Astley
(downloaded once per session via the rick_video fixture).
"""

import os

import pytest

from glean.chunker import (
    _compute_fixed_boundaries,
    _compute_smart_boundaries,
    _find_nearest_silence,
    detect_silence,
    get_duration,
    scan_directory,
)


class TestFixedBoundaries:
    def test_short_video(self):
        """Video shorter than chunk_duration gets one chunk."""
        bounds = _compute_fixed_boundaries(duration=20.0, chunk_duration=30, overlap=5)
        assert len(bounds) == 1
        assert bounds[0] == (0.0, 20.0)

    def test_exact_duration(self):
        """Video exactly chunk_duration gets one chunk."""
        bounds = _compute_fixed_boundaries(duration=30.0, chunk_duration=30, overlap=5)
        assert len(bounds) == 1
        assert bounds[0] == (0.0, 30.0)

    def test_two_chunks(self):
        bounds = _compute_fixed_boundaries(duration=50.0, chunk_duration=30, overlap=5)
        assert len(bounds) == 2
        assert bounds[0] == (0.0, 30.0)
        assert bounds[1] == (25.0, 50.0)

    def test_rick_duration(self):
        """~3:32 video should produce ~8 chunks at 30s/5s overlap."""
        bounds = _compute_fixed_boundaries(duration=212.0, chunk_duration=30, overlap=5)
        assert 7 <= len(bounds) <= 9
        # First chunk starts at 0
        assert bounds[0][0] == 0.0
        # Last chunk ends at or near duration
        assert bounds[-1][1] == 212.0

    def test_no_overlap(self):
        bounds = _compute_fixed_boundaries(duration=90.0, chunk_duration=30, overlap=0)
        assert len(bounds) == 3
        assert bounds[0] == (0.0, 30.0)
        assert bounds[1] == (30.0, 60.0)
        assert bounds[2] == (60.0, 90.0)


class TestFindNearestSilence:
    SILENCES = [
        {"start": 27.0, "end": 28.0, "midpoint": 27.5},
        {"start": 58.0, "end": 59.5, "midpoint": 58.75},
        {"start": 120.0, "end": 121.0, "midpoint": 120.5},
    ]

    def test_finds_nearby(self):
        result = _find_nearest_silence(30.0, self.SILENCES, search_window=5.0)
        assert result == 27.5

    def test_finds_closest(self):
        result = _find_nearest_silence(59.0, self.SILENCES, search_window=5.0)
        assert result == 58.75

    def test_none_when_too_far(self):
        result = _find_nearest_silence(90.0, self.SILENCES, search_window=5.0)
        assert result is None

    def test_empty_silences(self):
        result = _find_nearest_silence(30.0, [], search_window=5.0)
        assert result is None


class TestScanDirectory:
    def test_finds_videos(self, tmp_path):
        (tmp_path / "video.mp4").touch()
        (tmp_path / "video.mov").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.mkv").touch()

        files = scan_directory(str(tmp_path))
        extensions = {os.path.splitext(f)[1] for f in files}
        assert ".mp4" in extensions
        assert ".mov" in extensions
        assert ".mkv" in extensions
        assert ".txt" not in extensions
        assert len(files) == 3


# --- Integration tests (require rick_video fixture / network) ---

class TestRickIntegration:
    """Integration tests using the canonical Rick Astley test video.

    These tests download the video once per session and test real
    ffmpeg operations. They're slow but verify the full pipeline.
    """

    @pytest.mark.slow
    def test_get_duration(self, rick_video):
        duration = get_duration(rick_video)
        # Rick Astley is ~3:32 (212s), allow some variance
        assert 200 <= duration <= 220

    @pytest.mark.slow
    def test_detect_silence(self, rick_video):
        """Rick Astley has very few silences (continuous music)."""
        silences = detect_silence(rick_video)
        # Should find some gaps but not many -- it's a music video
        assert isinstance(silences, list)
        # Each silence should have the right structure
        for s in silences:
            assert "start" in s
            assert "end" in s
            assert "midpoint" in s
            assert s["start"] < s["end"]
            assert s["start"] <= s["midpoint"] <= s["end"]

    @pytest.mark.slow
    def test_smart_boundaries_fallback(self, rick_video):
        """With few silences, most boundaries should stay fixed."""
        duration = get_duration(rick_video)
        bounds = _compute_smart_boundaries(
            rick_video, duration,
            chunk_duration=30, min_chunk=5,
            search_window=5.0, verbose=False,
        )
        assert len(bounds) >= 6
        # First chunk starts at 0
        assert bounds[0][0] == 0.0
        # Last chunk ends at duration
        assert bounds[-1][1] == duration
