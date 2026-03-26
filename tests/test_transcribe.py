"""Tests for timestamp parsing and offset logic.

Uses Rick Astley lyrics as test data because of course it does.
"""

from glean.transcribe import (
    offset_timestamps,
    parse_timestamped_lines,
    _ts_to_seconds,
    _seconds_to_ts,
)


class TestTimestampConversion:
    def test_ts_to_seconds_zero(self):
        assert _ts_to_seconds("00", "00", "00.000") == 0.0

    def test_ts_to_seconds_minutes(self):
        assert _ts_to_seconds("00", "01", "30.000") == 90.0

    def test_ts_to_seconds_hours(self):
        assert _ts_to_seconds("01", "00", "00.000") == 3600.0

    def test_ts_to_seconds_fractional(self):
        assert _ts_to_seconds("00", "00", "05.320") == 5.32

    def test_seconds_to_ts_zero(self):
        assert _seconds_to_ts(0.0) == "00:00:00.000"

    def test_seconds_to_ts_minutes(self):
        assert _seconds_to_ts(90.0) == "00:01:30.000"

    def test_seconds_to_ts_hours(self):
        assert _seconds_to_ts(3661.5) == "01:01:01.500"

    def test_roundtrip(self):
        """Converting to seconds and back should preserve the value."""
        original = 125.750
        ts = _seconds_to_ts(original)
        parts = ts.split(":")
        result = _ts_to_seconds(parts[0], parts[1], parts[2])
        assert abs(result - original) < 0.001


class TestOffsetTimestamps:
    # Simulated Whisper output for a chunk starting at 0:00 in the chunk
    RICK_CHUNK_TRANSCRIPT = (
        "[00:00:00.000 --> 00:00:04.400]   We've known each other for so long\n"
        "[00:00:04.400 --> 00:00:09.200]   Your heart's been aching but you're too shy to say it\n"
        "[00:00:09.200 --> 00:00:12.640]   Inside we both know what's been going on"
    )

    def test_no_offset(self):
        """Zero offset should return unchanged text."""
        result = offset_timestamps(self.RICK_CHUNK_TRANSCRIPT, 0.0)
        assert result == self.RICK_CHUNK_TRANSCRIPT

    def test_offset_applied(self):
        """Offset of 60s should shift all timestamps by one minute."""
        result = offset_timestamps(self.RICK_CHUNK_TRANSCRIPT, 60.0)
        assert "[00:01:00.000 --> 00:01:04.400]" in result
        assert "[00:01:04.400 --> 00:01:09.200]" in result
        assert "We've known each other for so long" in result

    def test_offset_90s(self):
        """Chunk starting at 1:30 in the source video."""
        result = offset_timestamps(self.RICK_CHUNK_TRANSCRIPT, 90.0)
        assert "[00:01:30.000 --> 00:01:34.400]" in result

    def test_none_input(self):
        assert offset_timestamps(None, 60.0) is None

    def test_empty_input(self):
        assert offset_timestamps("", 60.0) == ""

    def test_non_timestamp_lines_preserved(self):
        """Lines without timestamps should pass through unchanged."""
        text = "Some header\n[00:00:05.000 --> 00:00:10.000]   Never gonna give you up"
        result = offset_timestamps(text, 30.0)
        assert "Some header" in result
        assert "[00:00:35.000 --> 00:00:40.000]" in result


class TestParseTimestampedLines:
    def test_parse_rick(self):
        text = (
            "[00:01:00.000 --> 00:01:04.400]   We've known each other for so long\n"
            "[00:01:04.400 --> 00:01:09.200]   Your heart's been aching but you're too shy to say it"
        )
        entries = parse_timestamped_lines(text)
        assert len(entries) == 2
        assert entries[0]["start"] == 60.0
        assert entries[0]["end"] == 64.4
        assert "known each other" in entries[0]["text"]
        assert entries[1]["start"] == 64.4

    def test_parse_empty(self):
        assert parse_timestamped_lines("") == []
        assert parse_timestamped_lines(None) == []
