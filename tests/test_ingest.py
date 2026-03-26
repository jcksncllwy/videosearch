"""Tests for the SRT parsing helpers used in YouTube ingest.

Test data comes from what YouTube actually serves for Never Gonna Give You Up.
"""

import os
import tempfile

from glean.ingest import _parse_srt, _parse_srt_timed, _slice_timed_captions


# Simulated YouTube auto-caption SRT for Rick Astley
RICK_SRT = """\
1
00:00:00,000 --> 00:00:03,500
<font color="#cccccc">♪ We're no strangers to love ♪</font>

2
00:00:03,500 --> 00:00:07,000
♪ You know the rules and so do I ♪

3
00:00:07,000 --> 00:00:11,500
♪ A full commitment's what I'm thinking of ♪

4
00:00:11,500 --> 00:00:15,000
♪ You wouldn't get this from any other guy ♪

5
00:00:15,000 --> 00:00:20,000
♪ I just wanna tell you how I'm feeling ♪

6
00:00:20,000 --> 00:00:24,000
♪ Gotta make you understand ♪

7
00:00:24,000 --> 00:00:27,500
♪ Never gonna give you up ♪

8
00:00:27,500 --> 00:00:31,000
♪ Never gonna let you down ♪

9
00:00:31,000 --> 00:00:35,000
♪ Never gonna run around and desert you ♪
"""


class TestParseSrt:
    def test_strips_html_tags(self):
        text = _parse_srt_from_string(RICK_SRT)
        assert "<font" not in text
        assert "strangers to love" in text

    def test_strips_sequence_numbers(self):
        text = _parse_srt_from_string(RICK_SRT)
        # Sequence numbers like "1", "2" on their own lines should be gone
        # (but numbers within lyrics should stay)
        assert text.startswith("\u266a") or text.startswith("We")

    def test_joins_lines(self):
        text = _parse_srt_from_string(RICK_SRT)
        # Should be a single string, not multiple lines
        assert "\n" not in text


class TestParseSrtTimed:
    def test_entry_count(self):
        entries = _parse_srt_timed_from_string(RICK_SRT)
        assert len(entries) == 9

    def test_timing(self):
        entries = _parse_srt_timed_from_string(RICK_SRT)
        assert entries[0]["start"] == 0.0
        assert entries[0]["end"] == 3.5
        assert entries[1]["start"] == 3.5

    def test_html_stripped(self):
        entries = _parse_srt_timed_from_string(RICK_SRT)
        assert "<font" not in entries[0]["text"]

    def test_text_content(self):
        entries = _parse_srt_timed_from_string(RICK_SRT)
        assert "strangers to love" in entries[0]["text"]
        assert "Never gonna give you up" in entries[6]["text"]


class TestSliceTimedCaptions:
    def test_slice_first_30s(self):
        entries = _parse_srt_timed_from_string(RICK_SRT)
        text = _slice_timed_captions(entries, 0.0, 30.0)
        assert text is not None
        assert "strangers to love" in text
        assert "Never gonna give you up" in text

    def test_slice_excludes_out_of_range(self):
        entries = _parse_srt_timed_from_string(RICK_SRT)
        # Only get entries from 25-35s
        text = _slice_timed_captions(entries, 25.0, 36.0)
        assert text is not None
        assert "Never gonna give you up" in text
        assert "Never gonna let you down" in text
        # Earlier lyrics shouldn't be in this slice
        assert "strangers to love" not in text

    def test_slice_empty_range(self):
        entries = _parse_srt_timed_from_string(RICK_SRT)
        text = _slice_timed_captions(entries, 100.0, 130.0)
        assert text is None


# --- Helpers to parse from string instead of file ---

def _parse_srt_from_string(srt_content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
        f.write(srt_content)
        f.flush()
        try:
            return _parse_srt(f.name)
        finally:
            os.unlink(f.name)


def _parse_srt_timed_from_string(srt_content: str) -> list[dict]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
        f.write(srt_content)
        f.flush()
        try:
            return _parse_srt_timed(f.name)
        finally:
            os.unlink(f.name)
