"""Extract clips from source videos."""

import os
import re
import subprocess

from .chunker import get_ffmpeg, get_duration


def trim_clip(
    source_file: str,
    start_time: float,
    end_time: float,
    output_path: str,
    padding: float = 2.0,
) -> str:
    """Extract a segment from a video with padding.

    Tries stream copy first (fast), falls back to re-encode.
    """
    if end_time <= start_time:
        raise ValueError(f"end_time ({end_time}) must be > start_time ({start_time})")

    duration = get_duration(source_file)
    padded_start = max(0.0, start_time - padding)
    padded_end = min(duration, end_time + padding)
    length = padded_end - padded_start

    ffmpeg = get_ffmpeg()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Try stream copy first
    subprocess.run(
        [ffmpeg, "-y", "-ss", str(padded_start), "-i", source_file,
         "-t", str(length), "-c", "copy", output_path],
        capture_output=True,
    )
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 1024:
        return output_path

    # Fall back to re-encode
    subprocess.run(
        [ffmpeg, "-y", "-i", source_file, "-ss", str(padded_start),
         "-t", str(length), "-c:v", "libx264", "-crf", "23",
         "-c:a", "aac", "-b:a", "128k", output_path],
        capture_output=True, check=True,
    )
    return output_path


def fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def safe_filename(source_file: str, start: float, end: float) -> str:
    base = os.path.splitext(os.path.basename(source_file))[0]
    base = re.sub(r"[^\w\-]", "_", base)
    s = fmt_time(start).replace(":", "m") + "s"
    e = fmt_time(end).replace(":", "m") + "s"
    return f"clip_{base}_{s}-{e}.mp4"
