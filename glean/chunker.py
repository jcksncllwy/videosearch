"""Video chunking and scanning.

Supports two chunking modes:
- Fixed interval (default): chunks at regular intervals with overlap
- Smart (silence-based): detects silence gaps in audio and snaps chunk
  boundaries to natural pauses, so sentences/phrases don't get split
"""

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


# ------------------------------------------------------------------
# Silence detection
# ------------------------------------------------------------------

def detect_silence(
    video_path: str,
    noise_threshold: str = "-30dB",
    min_silence_duration: float = 0.3,
) -> list[dict]:
    """Detect silence gaps in a video's audio track using ffmpeg.

    Args:
        video_path: Path to video file.
        noise_threshold: Volume below which is considered silence (default -30dB).
        min_silence_duration: Minimum silence length in seconds to count as a gap.

    Returns:
        List of {start, end, midpoint} dicts for each silence gap.
    """
    ffmpeg = get_ffmpeg()
    result = subprocess.run(
        [
            ffmpeg, "-i", video_path,
            "-af", f"silencedetect=noise={noise_threshold}:d={min_silence_duration}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, check=False,
    )

    silences = []
    current_start = None

    for line in result.stderr.split("\n"):
        # silencedetect outputs lines like:
        #   [silencedetect @ 0x...] silence_start: 12.345
        #   [silencedetect @ 0x...] silence_end: 13.456 | silence_duration: 1.111
        start_match = re.search(r"silence_start:\s*(\d+\.?\d*)", line)
        end_match = re.search(r"silence_end:\s*(\d+\.?\d*)", line)

        if start_match:
            current_start = float(start_match.group(1))
        elif end_match and current_start is not None:
            end = float(end_match.group(1))
            midpoint = (current_start + end) / 2
            silences.append({
                "start": current_start,
                "end": end,
                "midpoint": midpoint,
            })
            current_start = None

    return silences


def _find_nearest_silence(
    target_time: float,
    silences: list[dict],
    search_window: float = 5.0,
) -> float | None:
    """Find the silence midpoint nearest to target_time within search_window.

    Returns the midpoint time, or None if no silence found nearby.
    """
    best = None
    best_dist = float("inf")

    for s in silences:
        dist = abs(s["midpoint"] - target_time)
        if dist <= search_window and dist < best_dist:
            best = s["midpoint"]
            best_dist = dist

    return best


# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------

def chunk_video(
    video_path: str,
    chunk_duration: int = 30,
    overlap: int = 5,
    keep_audio: bool = True,
    smart: bool = False,
    silence_search_window: float = 5.0,
    verbose: bool = False,
) -> list[dict]:
    """Split video into chunks. Returns list of chunk dicts.

    Each chunk dict: {chunk_path, source_file, start_time, end_time}

    Args:
        video_path: Path to video file.
        chunk_duration: Target duration per chunk in seconds.
        overlap: Overlap between chunks (fixed mode) or minimum gap (smart mode).
        keep_audio: Preserve audio track in chunks.
        smart: If True, snap chunk boundaries to silence gaps in the audio.
            Falls back to fixed intervals where no silence is found.
        silence_search_window: How far (seconds) from the target boundary to
            search for a silence gap. Only used in smart mode.
        verbose: Print chunking decisions to stderr.
    """
    video_path = str(Path(video_path).resolve())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = get_duration(video_path)

    if smart:
        boundaries = _compute_smart_boundaries(
            video_path, duration, chunk_duration, overlap,
            silence_search_window, verbose,
        )
    else:
        boundaries = _compute_fixed_boundaries(duration, chunk_duration, overlap)

    return _cut_chunks(video_path, boundaries, keep_audio)


def _compute_fixed_boundaries(
    duration: float, chunk_duration: int, overlap: int,
) -> list[tuple[float, float]]:
    """Compute fixed-interval chunk boundaries with overlap."""
    step = max(1, chunk_duration - overlap)
    boundaries = []
    start = 0.0

    while start < duration:
        end = min(start + chunk_duration, duration)
        if boundaries and (end - start) <= overlap:
            break
        boundaries.append((start, end))
        start += step

    return boundaries


def _compute_smart_boundaries(
    video_path: str,
    duration: float,
    chunk_duration: int,
    min_chunk: int,
    search_window: float,
    verbose: bool,
) -> list[tuple[float, float]]:
    """Compute chunk boundaries snapped to silence gaps.

    Strategy:
    1. Detect all silence gaps in the audio
    2. Walk through the video at chunk_duration intervals
    3. At each boundary, look for a nearby silence gap
    4. If found, snap the boundary to the silence midpoint
    5. If not found, use the fixed boundary (don't split mid-word,
       but don't create a huge chunk either)

    Chunks won't be shorter than min_chunk seconds or longer than
    chunk_duration + search_window seconds.
    """
    import sys

    silences = detect_silence(video_path)

    if verbose:
        print(f"    [chunk] Detected {len(silences)} silence gaps", file=sys.stderr)

    if not silences:
        if verbose:
            print("    [chunk] No silence detected, using fixed intervals", file=sys.stderr)
        return _compute_fixed_boundaries(duration, chunk_duration, min_chunk)

    boundaries = []
    start = 0.0

    while start < duration:
        target_end = start + chunk_duration

        if target_end >= duration:
            # Last chunk -- take whatever's left
            boundaries.append((start, duration))
            break

        # Look for silence near the target boundary
        snap = _find_nearest_silence(target_end, silences, search_window)

        if snap is not None and snap > start + min_chunk:
            end = snap
            if verbose:
                print(
                    f"    [chunk] Snapped {target_end:.1f}s -> {end:.1f}s (silence)",
                    file=sys.stderr,
                )
        else:
            end = target_end
            if verbose:
                print(
                    f"    [chunk] No silence near {target_end:.1f}s, using fixed boundary",
                    file=sys.stderr,
                )

        boundaries.append((start, end))
        start = end  # No overlap in smart mode -- boundaries are clean breaks

    return boundaries


def _cut_chunks(
    video_path: str,
    boundaries: list[tuple[float, float]],
    keep_audio: bool,
) -> list[dict]:
    """Cut video at the given boundaries."""
    ffmpeg = get_ffmpeg()
    tmp_dir = tempfile.mkdtemp(prefix="glean_")
    chunks = []

    for idx, (start, end) in enumerate(boundaries):
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

    return chunks


def extract_audio(video_path: str, output_path: str | None = None) -> str:
    """Extract audio from a video file as 16kHz mono WAV (Whisper-ready)."""
    if output_path is None:
        # NamedTemporaryFile with delete=False instead of deprecated mktemp
        f = tempfile.NamedTemporaryFile(prefix="glean_audio_", suffix=".wav", delete=False)
        output_path = f.name
        f.close()
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
