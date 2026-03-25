"""Video chunking and scanning."""

import functools
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


@functools.lru_cache(maxsize=1)
def get_ffmpeg() -> str:
    """Return a usable ffmpeg path. Prefers system, falls back to imageio-ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found. Install it (brew install ffmpeg) or pip install imageio-ffmpeg."
        ) from exc


def get_ffprobe() -> str | None:
    """Return ffprobe path if available."""
    return shutil.which("ffprobe")


def get_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    ffprobe = get_ffprobe()
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])

    # Fallback: parse ffmpeg stderr
    result = subprocess.run(
        [get_ffmpeg(), "-i", video_path], capture_output=True, text=True, check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError(f"Could not determine duration of {video_path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def chunk_video(
    video_path: str,
    chunk_duration: int = 30,
    overlap: int = 5,
    keep_audio: bool = True,
) -> list[dict]:
    """Split video into overlapping chunks. Returns list of chunk dicts.

    Each chunk dict: {chunk_path, source_file, start_time, end_time}
    Audio is preserved by default (unlike SentrySearch which strips it).
    """
    video_path = str(Path(video_path).resolve())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffmpeg = get_ffmpeg()
    duration = get_duration(video_path)
    tmp_dir = tempfile.mkdtemp(prefix="videosearch_")
    step = max(1, chunk_duration - overlap)
    chunks = []

    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk_duration, duration)
        # Skip tiny trailing chunks
        if idx > 0 and (end - start) < overlap:
            break
        chunk_path = os.path.join(tmp_dir, f"chunk_{idx:04d}.mp4")
        cmd = [
            ffmpeg, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(end - start),
            "-c", "copy",
        ]
        if not keep_audio:
            cmd += ["-an"]
        cmd.append(chunk_path)
        subprocess.run(cmd, capture_output=True, check=True)

        chunks.append({
            "chunk_path": chunk_path,
            "source_file": video_path,
            "start_time": start,
            "end_time": end,
        })
        start += step
        idx += 1

    return chunks


def extract_audio(video_path: str, output_path: str | None = None) -> str:
    """Extract audio from a video file as 16kHz mono WAV (Whisper-ready)."""
    if output_path is None:
        output_path = tempfile.mktemp(prefix="videosearch_audio_", suffix=".wav")
    subprocess.run(
        [get_ffmpeg(), "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", output_path],
        capture_output=True, check=True,
    )
    return output_path


def scan_directory(directory: str) -> list[str]:
    """Recursively find all video files in a directory."""
    files = []
    for root, _dirs, names in os.walk(directory):
        for name in names:
            if Path(name).suffix.lower() in VIDEO_EXTENSIONS:
                files.append(os.path.join(root, name))
    files.sort()
    return files
