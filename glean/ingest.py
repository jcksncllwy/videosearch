"""Ingest pipelines for different video sources.

Each pipeline: download (if needed) -> chunk -> transcribe -> extract entities.
Entity extraction runs automatically via Claude Sonnet, creating/updating
structured notes in the Obsidian vault knowledge graph.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .chunker import chunk_video, extract_audio, get_duration, scan_directory
from .store import VideoStore
from .transcribe import transcribe_video_chunk

DEFAULT_COOKIE_JAR = Path.home() / ".glean" / "youtube-cookies.txt"

# yt-dlp throttle args -- applied to all yt-dlp calls to avoid triggering rate limits
YTDLP_THROTTLE_ARGS = [
    "--sleep-requests", "1",       # 1s between API requests
    "--sleep-interval", "5",       # 5-10s before each download
    "--max-sleep-interval", "10",
    "--retry-sleep", "http:5",     # 5s before retrying HTTP errors
]
THROTTLE_FILE = Path.home() / ".glean" / "throttle.json"

# Minimum seconds between Instagram API requests
INSTAGRAM_MIN_INTERVAL = 10
# Backoff after a rate limit error (seconds)
INSTAGRAM_RATE_LIMIT_BACKOFF = 300  # 5 minutes


def _check_instagram_throttle() -> float:
    """Check throttle state. Returns seconds to wait, or 0 if clear."""
    if not THROTTLE_FILE.exists():
        return 0
    try:
        data = json.loads(THROTTLE_FILE.read_text())
        ig = data.get("instagram", {})

        # Check if we're in a rate-limit backoff period
        blocked_until = ig.get("blocked_until", 0)
        if blocked_until > time.time():
            return blocked_until - time.time()

        # Check minimum interval between requests
        last_request = ig.get("last_request", 0)
        elapsed = time.time() - last_request
        if elapsed < INSTAGRAM_MIN_INTERVAL:
            return INSTAGRAM_MIN_INTERVAL - elapsed

        return 0
    except (json.JSONDecodeError, KeyError):
        return 0


def _record_instagram_request():
    """Record that we just made an Instagram API request."""
    data = {}
    if THROTTLE_FILE.exists():
        try:
            data = json.loads(THROTTLE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    ig = data.get("instagram", {})
    ig["last_request"] = time.time()
    data["instagram"] = ig
    THROTTLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    THROTTLE_FILE.write_text(json.dumps(data, indent=2))


def _record_instagram_rate_limit():
    """Record that we hit a rate limit — enforce a backoff period."""
    data = {}
    if THROTTLE_FILE.exists():
        try:
            data = json.loads(THROTTLE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    ig = data.get("instagram", {})
    ig["blocked_until"] = time.time() + INSTAGRAM_RATE_LIMIT_BACKOFF
    ig["last_rate_limit"] = time.time()
    data["instagram"] = ig
    THROTTLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    THROTTLE_FILE.write_text(json.dumps(data, indent=2))


def _ytdlp_cookie_args(cookies_from_browser: str | None = None) -> list[str]:
    """Build yt-dlp cookie arguments.

    Prefers a saved cookie jar at ~/.glean/youtube-cookies.txt (no
    Keychain popup). Falls back to --cookies-from-browser if specified.
    """
    if DEFAULT_COOKIE_JAR.exists():
        return ["--cookies", str(DEFAULT_COOKIE_JAR)]
    if cookies_from_browser:
        return ["--cookies-from-browser", cookies_from_browser]
    return []


def ingest_local(
    path: str,
    store: VideoStore,
    chunk_duration: int = 30,
    overlap: int = 5,
    smart_chunks: bool = True,
    whisper_model: str | None = None,
    extract_entities: bool = True,
    verbose: bool = False,
    on_progress=None,
) -> dict:
    """Ingest local video files. Chunks and transcribes (free).

    Args:
        path: File or directory path.
        store: VideoStore instance.
        chunk_duration: Seconds per chunk.
        overlap: Overlap between chunks in seconds.
        smart_chunks: Snap chunk boundaries to silence gaps.
        whisper_model: Whisper model name.
        verbose: Print debug info.
        on_progress: Callback(message: str) for progress updates.

    Returns:
        Summary dict with counts.
    """
    if os.path.isfile(path):
        files = [os.path.abspath(path)]
    else:
        files = scan_directory(path)

    if not files:
        return {"files": 0, "chunks": 0, "transcribed": 0, "skipped": 0}

    total_files = 0
    total_chunks = 0
    total_transcribed = 0
    total_skipped = 0

    for file_path in files:
        abs_path = os.path.abspath(file_path)
        basename = os.path.basename(file_path)

        if store.is_video_indexed(abs_path):
            if on_progress:
                on_progress(f"Skipping {basename} (already indexed)")
            total_skipped += 1
            continue

        if on_progress:
            on_progress(f"Ingesting {basename}...")

        duration = get_duration(abs_path)
        video_id = store.add_video(
            path=abs_path,
            source_type="local",
            title=basename,
            duration=duration,
        )

        chunks = chunk_video(
            abs_path,
            chunk_duration=chunk_duration,
            overlap=overlap,
            keep_audio=True,
            smart=smart_chunks,
            verbose=verbose,
        )

        chunk_transcripts = []
        for i, chunk in enumerate(chunks):
            if on_progress:
                on_progress(f"  {basename} chunk {i + 1}/{len(chunks)}: transcribing...")

            plain, timestamped = transcribe_video_chunk(
                chunk["chunk_path"],
                chunk_start_time=chunk["start_time"],
                model=whisper_model,
            )

            store.add_chunk(
                video_id=video_id,
                start_time=chunk["start_time"],
                end_time=chunk["end_time"],
                chunk_path=chunk["chunk_path"],
                transcript=plain,
                transcript_timestamped=timestamped,
                transcript_source="whisper",
            )

            chunk_transcripts.append({
                "transcript": plain,
                "transcript_timestamped": timestamped,
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
            })

            if plain:
                total_transcribed += 1
            total_chunks += 1

        # Entity extraction
        if extract_entities and chunk_transcripts:
            _run_extraction(
                chunk_transcripts, title=basename, source_type="local",
                duration=duration, verbose=verbose, on_progress=on_progress,
            )

        total_files += 1

    return {
        "files": total_files,
        "chunks": total_chunks,
        "transcribed": total_transcribed,
        "skipped": total_skipped,
    }


def ingest_youtube(
    url: str,
    store: VideoStore,
    chunk_duration: int = 30,
    overlap: int = 5,
    smart_chunks: bool = True,
    whisper_model: str | None = None,
    extract_entities: bool = True,
    cookies_from_browser: str | None = None,
    verbose: bool = False,
    on_progress=None,
) -> dict:
    """Download and ingest a YouTube video.

    Grabs both the video and any available captions. Runs Whisper too
    (captions are often mediocre). Both transcript sources are kept.
    YouTube SRT captions are time-sliced per chunk so each chunk gets
    only the captions relevant to its time range.
    """
    import json as json_mod

    # Get video metadata first
    cookie_args = _ytdlp_cookie_args(cookies_from_browser)
    meta_cmd = ["yt-dlp", "--dump-json", "--no-download"] + YTDLP_THROTTLE_ARGS + cookie_args + [url]
    meta_result = subprocess.run(meta_cmd, capture_output=True, text=True)
    if meta_result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {meta_result.stderr}")
    meta = json_mod.loads(meta_result.stdout)
    title = meta.get("title", "Unknown")
    duration = meta.get("duration")
    description = meta.get("description") or ""
    channel = meta.get("channel") or meta.get("uploader") or ""

    if on_progress:
        on_progress(f"Downloading: {title}")
        if channel:
            on_progress(f"  Channel: {channel}")

    # Download video + subtitles to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="glean_yt_")
    output_template = os.path.join(tmp_dir, "%(id)s.%(ext)s")

    dl_cmd = [
        "yt-dlp",
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--write-subs", "--write-auto-subs",
        "--sub-lang", "en",
        "--convert-subs", "srt",
        "-o", output_template,
    ] + YTDLP_THROTTLE_ARGS + cookie_args + [url]
    dl_result = subprocess.run(dl_cmd, capture_output=True, text=True)
    if dl_result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {dl_result.stderr}")

    # Find the downloaded video file
    video_file = None
    srt_file = None
    for f in os.listdir(tmp_dir):
        if f.endswith(".mp4"):
            video_file = os.path.join(tmp_dir, f)
        elif f.endswith(".srt"):
            srt_file = os.path.join(tmp_dir, f)

    if not video_file:
        raise RuntimeError("No video file found after download")

    # Parse YouTube captions if available (structured with timestamps)
    yt_timed_entries = None
    if srt_file:
        yt_timed_entries = _parse_srt_timed(srt_file)
        if on_progress:
            on_progress(f"  Found YouTube captions ({len(yt_timed_entries)} entries)")

    # Register video
    video_id = store.add_video(
        path=video_file,
        source_type="youtube",
        source_url=url,
        title=title,
        description=description,
        channel=channel,
        duration=duration,
    )

    # Chunk and transcribe
    chunks = chunk_video(
        video_file,
        chunk_duration=chunk_duration,
        overlap=overlap,
        keep_audio=True,
        smart=smart_chunks,
        verbose=verbose,
    )
    total_transcribed = 0
    chunk_transcripts = []

    for i, chunk in enumerate(chunks):
        if on_progress:
            on_progress(f"  Chunk {i + 1}/{len(chunks)}: transcribing...")

        # Get Whisper transcript (with absolute timestamps)
        whisper_plain, whisper_ts = transcribe_video_chunk(
            chunk["chunk_path"],
            chunk_start_time=chunk["start_time"],
            model=whisper_model,
        )

        # Get YouTube caption slice for this chunk's time range
        yt_slice = None
        if yt_timed_entries:
            yt_slice = _slice_timed_captions(
                yt_timed_entries, chunk["start_time"], chunk["end_time"]
            )

        # Prefer Whisper for transcript field (generally better quality),
        # but store both. Use YouTube captions as fallback if Whisper fails.
        transcript = whisper_plain or yt_slice
        # Combine both sources for better search coverage
        if whisper_plain and yt_slice and whisper_plain != yt_slice:
            transcript = f"{whisper_plain}\n---\n{yt_slice}"

        source = "whisper"
        if not whisper_plain and yt_slice:
            source = "youtube_captions"
        elif whisper_plain and yt_slice:
            source = "whisper+youtube"

        store.add_chunk(
            video_id=video_id,
            start_time=chunk["start_time"],
            end_time=chunk["end_time"],
            chunk_path=chunk["chunk_path"],
            transcript=transcript,
            transcript_timestamped=whisper_ts,
            transcript_source=source,
        )

        chunk_transcripts.append({
            "transcript": transcript,
            "transcript_timestamped": whisper_ts,
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
        })

        if transcript:
            total_transcribed += 1

    # Entity extraction
    if extract_entities and chunk_transcripts:
        _run_extraction(
            chunk_transcripts, title=title, source_type="youtube",
            url=url, duration=duration, description=description,
            channel=channel, verbose=verbose, on_progress=on_progress,
        )

    # Clean up downloaded source video (chunks are separate temp files)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "files": 1,
        "chunks": len(chunks),
        "transcribed": total_transcribed,
        "skipped": 0,
        "title": title,
        "has_youtube_captions": srt_file is not None,
    }


def ingest_instagram(
    url: str,
    store: VideoStore,
    whisper_model: str | None = None,
    extract_entities: bool = True,
    verbose: bool = False,
    on_progress=None,
) -> dict:
    """Download and ingest an Instagram reel.

    Reels are short (15-90s) so typically a single chunk, no splitting needed.
    Captures the post caption as description for search and entity extraction.
    Uses instaloader for downloading (no auth needed for public posts).
    """
    import re as re_mod
    import instaloader

    # Extract shortcode from URL
    match = re_mod.search(r'/(?:reel|reels|p)/([A-Za-z0-9_-]+)', url)
    if not match:
        raise ValueError(f"Could not extract shortcode from URL: {url}")
    shortcode = match.group(1)

    # Check throttle before making any Instagram API calls
    wait_time = _check_instagram_throttle()
    if wait_time > 0:
        mins = int(wait_time // 60)
        secs = int(wait_time % 60)
        raise RuntimeError(
            f"Instagram rate limit cooldown: {mins}m {secs}s remaining. "
            f"Try again later or check ~/.glean/throttle.json"
        )

    if on_progress:
        on_progress("Downloading reel via instaloader...")

    tmp_dir = tempfile.mkdtemp(prefix="glean_ig_")

    L = instaloader.Instaloader(
        dirname_pattern=tmp_dir,
        filename_pattern="{shortcode}",
        download_videos=True,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
    )

    # Keep instaloader's native rate controller (handles pacing and 429 backoff).
    # Cap retries at 3 so we don't hang forever on non-429 errors (401/403),
    # but let the RateController's sleep/wait logic work normally.
    L.context.max_connection_attempts = 3

    # Record the request for throttling
    _record_instagram_request()

    # Fetch the post
    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "429" in err_str or "rate" in err_str.lower():
            _record_instagram_rate_limit()
            raise RuntimeError(
                f"Instagram rate limited. Backing off for "
                f"{INSTAGRAM_RATE_LIMIT_BACKOFF // 60} minutes. Error: {e}"
            )
        raise

    description = post.caption or ""
    channel = post.owner_username or ""
    meta_title = post.title or ""

    if on_progress and channel:
        on_progress(f"  Creator: {channel}")

    # Download the video
    _record_instagram_request()
    L.download_post(post, target="")

    # Find downloaded video file
    video_file = None
    for f in os.listdir(tmp_dir):
        if f.endswith((".mp4", ".webm", ".mkv")):
            video_file = os.path.join(tmp_dir, f)
            break

    if not video_file:
        raise RuntimeError("No video file found after download — post may not be a video")

    duration = get_duration(video_file)
    # Use metadata title if available, fall back to filename
    title = meta_title or os.path.splitext(os.path.basename(video_file))[0]

    video_id = store.add_video(
        path=video_file,
        source_type="instagram",
        source_url=url,
        title=title,
        description=description,
        channel=channel,
        duration=duration,
    )

    # Use chunk_video for consistent chunk dict structure. Short reels
    # get one chunk; longer ones get smart-split like YouTube videos.
    chunks = chunk_video(
        video_file,
        chunk_duration=max(30, int(duration) + 1),  # single chunk for short reels
        overlap=5, keep_audio=True, smart=True, verbose=verbose,
    )

    total_transcribed = 0
    chunk_transcripts = []
    for i, chunk in enumerate(chunks):
        if on_progress:
            on_progress(f"  Chunk {i + 1}/{len(chunks)}: transcribing...")

        plain, timestamped = transcribe_video_chunk(
            chunk["chunk_path"],
            chunk_start_time=chunk["start_time"],
            model=whisper_model,
        )
        store.add_chunk(
            video_id=video_id,
            start_time=chunk["start_time"],
            end_time=chunk["end_time"],
            chunk_path=chunk["chunk_path"],
            transcript=plain,
            transcript_timestamped=timestamped,
            transcript_source="whisper",
        )
        chunk_transcripts.append({
            "transcript": plain,
            "transcript_timestamped": timestamped,
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
        })
        if plain:
            total_transcribed += 1

    # Entity extraction
    if extract_entities and chunk_transcripts:
        _run_extraction(
            chunk_transcripts, title=title, source_type="instagram",
            url=url, duration=duration, description=description,
            channel=channel, verbose=verbose, on_progress=on_progress,
        )

    # Clean up downloaded source video
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "files": 1,
        "chunks": len(chunks),
        "transcribed": total_transcribed,
        "skipped": 0,
        "title": title,
    }


# ------------------------------------------------------------------
# Entity extraction helper
# ------------------------------------------------------------------

def _run_extraction(
    chunk_transcripts: list[dict],
    title: str,
    source_type: str,
    url: str | None = None,
    duration: float | None = None,
    description: str | None = None,
    channel: str | None = None,
    verbose: bool = False,
    on_progress=None,
):
    """Run entity extraction and persist to vault. Fails silently."""
    try:
        from .extract import extract_and_persist
        extract_and_persist(
            chunk_transcripts, title=title, source_type=source_type,
            url=url, duration=duration, description=description,
            channel=channel, verbose=verbose, on_progress=on_progress,
        )
    except Exception as e:
        if on_progress:
            on_progress(f"  Entity extraction failed (non-fatal): {e}")


# ------------------------------------------------------------------
# SRT parsing helpers
# ------------------------------------------------------------------

def _parse_srt(srt_path: str) -> str:
    """Parse an SRT file into plain text."""
    import re
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Strip SRT formatting: sequence numbers, timestamps, blank lines
    lines = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+$", line):
            continue
        if re.match(r"\d{2}:\d{2}:\d{2}", line):
            continue
        # Strip HTML tags sometimes in YouTube SRTs
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)

    return " ".join(lines)


def _parse_srt_timed(srt_path: str) -> list[dict]:
    """Parse SRT into list of {start, end, text} dicts."""
    import re
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    entries = []
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1] if len(lines) > 1 else "",
        )
        if not time_match:
            continue
        g = time_match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
        text = " ".join(lines[2:])
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            entries.append({"start": start, "end": end, "text": text})

    return entries


def _slice_timed_captions(
    entries: list[dict], start: float, end: float,
) -> str | None:
    """Extract caption text that overlaps with a time range.

    Args:
        entries: List of {start, end, text} from _parse_srt_timed.
        start: Chunk start time in seconds.
        end: Chunk end time in seconds.

    Returns:
        Concatenated caption text for the range, or None if empty.
    """
    lines = []
    for entry in entries:
        # Include if the caption overlaps with the chunk at all
        if entry["end"] > start and entry["start"] < end:
            lines.append(entry["text"])

    return " ".join(lines) if lines else None
