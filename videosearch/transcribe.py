"""Transcription via whisper.cpp (local, free).

Uses the `transcribe` script from ~/ygg/bin/ which wraps whisper-cli.
Falls back to a direct whisper-cli call if the script isn't found.

Timestamps are offset to absolute video time so each line maps directly
to a point in the source video, not relative to the chunk start.
"""

import os
import re
import subprocess
from pathlib import Path

from .chunker import extract_audio

TRANSCRIBE_BIN = os.path.join(os.environ.get("HOME", ""), "ygg", "bin", "transcribe")
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

# Matches Whisper timestamp lines like: [00:00:05.000 --> 00:00:08.320]   some text
_TS_LINE_RE = re.compile(
    r"\[(\d{2}):(\d{2}):(\d{2}\.\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}\.\d{3})\](.*)"
)


def _has_transcribe_script() -> bool:
    return os.path.isfile(TRANSCRIBE_BIN) and os.access(TRANSCRIBE_BIN, os.X_OK)


def _ts_to_seconds(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


def _seconds_to_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def offset_timestamps(timestamped_text: str, offset_seconds: float) -> str:
    """Shift all Whisper timestamps by offset_seconds.

    Converts chunk-relative timestamps to absolute video timestamps.
    E.g., if chunk starts at 90s in the source video, [00:00:05.000]
    becomes [00:01:35.000].
    """
    if not timestamped_text or offset_seconds == 0:
        return timestamped_text

    lines = []
    for line in timestamped_text.split("\n"):
        match = _TS_LINE_RE.match(line)
        if match:
            g = match.groups()
            start = _ts_to_seconds(g[0], g[1], g[2]) + offset_seconds
            end = _ts_to_seconds(g[3], g[4], g[5]) + offset_seconds
            text = g[6]
            lines.append(f"[{_seconds_to_ts(start)} --> {_seconds_to_ts(end)}]{text}")
        else:
            lines.append(line)

    return "\n".join(lines)


def parse_timestamped_lines(timestamped_text: str) -> list[dict]:
    """Parse timestamped transcript into structured list.

    Returns list of {start, end, text} dicts. Timestamps should already
    be absolute (offset applied) so they map directly to the source video.
    """
    if not timestamped_text:
        return []

    entries = []
    for line in timestamped_text.split("\n"):
        match = _TS_LINE_RE.match(line)
        if match:
            g = match.groups()
            entries.append({
                "start": _ts_to_seconds(g[0], g[1], g[2]),
                "end": _ts_to_seconds(g[3], g[4], g[5]),
                "text": g[6].strip(),
            })

    return entries


def transcribe_file(
    audio_path: str,
    model: str | None = None,
    timestamps: bool = True,
) -> str | None:
    """Transcribe an audio/video file to text.

    Args:
        audio_path: Path to audio or video file.
        model: Whisper model name (default: large-v3-turbo).
        timestamps: If True, include timestamps in output.

    Returns:
        Transcribed text, or None on failure.
    """
    model = model or DEFAULT_MODEL

    if _has_transcribe_script():
        cmd = [TRANSCRIBE_BIN, "-m", model]
        if timestamps:
            cmd.append("-t")
        cmd.append(audio_path)
    else:
        # Direct whisper-cli fallback
        whisper = _find_whisper_cli()
        if not whisper:
            return None
        models_dir = os.path.expanduser("~/.local/share/whisper-cpp/models")
        model_path = os.path.join(models_dir, f"ggml-{model}.bin")
        if not os.path.isfile(model_path):
            return None
        cmd = [whisper, "-m", model_path, "-f", audio_path, "-np"]
        if not timestamps:
            cmd.append("--no-timestamps")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env={**os.environ, "PATH": f"{os.environ.get('HOME', '')}/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:{os.environ.get('PATH', '')}"},
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def transcribe_video_chunk(
    chunk_path: str,
    chunk_start_time: float = 0.0,
    model: str | None = None,
) -> tuple[str | None, str | None]:
    """Transcribe a video chunk. Extracts audio first.

    Args:
        chunk_path: Path to the video chunk file.
        chunk_start_time: When this chunk starts in the source video (seconds).
            Used to offset Whisper's timestamps to absolute video time.
        model: Whisper model name.

    Returns:
        (plain_text, timestamped_text) -- timestamps are absolute (offset applied).
        Either may be None on failure.
    """
    # Extract audio to WAV
    try:
        wav_path = extract_audio(chunk_path)
    except Exception:
        return None, None

    try:
        plain = transcribe_file(wav_path, model=model, timestamps=False)
        timestamped = transcribe_file(wav_path, model=model, timestamps=True)

        # Offset timestamps to absolute video time
        if timestamped and chunk_start_time > 0:
            timestamped = offset_timestamps(timestamped, chunk_start_time)

        return plain, timestamped
    finally:
        # Clean up temp WAV
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _find_whisper_cli() -> str | None:
    """Find whisper-cli binary."""
    import shutil
    for name in ["whisper-cli", "whisper-cpp"]:
        path = shutil.which(name)
        if path:
            return path
    for loc in ["/opt/homebrew/bin/whisper-cli", "/usr/local/bin/whisper-cli"]:
        if os.path.isfile(loc):
            return loc
    return None
