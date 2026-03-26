"""CLI entry point for VideoSearch."""

import os
import platform
import subprocess

import click
from dotenv import load_dotenv

load_dotenv()


def _fmt_time(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _open_file(path: str):
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


@click.group()
def cli():
    """Search video collections with natural language.

    Ingest videos from local files, YouTube, or Instagram.
    Search uses free transcript lookup first, with optional paid visual search.
    """


# -----------------------------------------------------------------------
# ingest
# -----------------------------------------------------------------------

@cli.group()
def ingest():
    """Ingest videos from various sources."""


@ingest.command("local")
@click.argument("path", type=click.Path(exists=True))
@click.option("--chunk-duration", default=30, show_default=True, help="Target chunk duration in seconds.")
@click.option("--overlap", default=5, show_default=True, help="Overlap between chunks (fixed mode) or min chunk size (smart mode).")
@click.option("--smart/--no-smart", default=True, show_default=True,
              help="Snap chunk boundaries to silence gaps in audio.")
@click.option("--extract/--no-extract", default=True, show_default=True,
              help="Extract entities to Obsidian vault after transcription.")
@click.option("--whisper-model", default=None, help="Whisper model (default: large-v3-turbo).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def ingest_local_cmd(path, chunk_duration, overlap, smart, extract, whisper_model, verbose):
    """Ingest local video files from PATH (file or directory)."""
    from .ingest import ingest_local
    from .store import VideoStore

    store = VideoStore()

    def progress(msg):
        click.echo(msg)

    result = ingest_local(
        path, store,
        chunk_duration=chunk_duration,
        overlap=overlap,
        smart_chunks=smart,
        whisper_model=whisper_model,
        extract_entities=extract,
        verbose=verbose,
        on_progress=progress,
    )

    click.echo(
        f"\nIngested {result['chunks']} chunks from {result['files']} files. "
        f"Transcribed: {result['transcribed']}. Skipped: {result['skipped']}."
    )
    store.close()


@ingest.command("youtube")
@click.argument("url")
@click.option("--chunk-duration", default=30, show_default=True, help="Target chunk duration in seconds.")
@click.option("--overlap", default=5, show_default=True, help="Overlap/min chunk size in seconds.")
@click.option("--smart/--no-smart", default=True, show_default=True,
              help="Snap chunk boundaries to silence gaps in audio.")
@click.option("--extract/--no-extract", default=True, show_default=True,
              help="Extract entities to Obsidian vault after transcription.")
@click.option("--cookies-from", default=None, help="Browser to extract cookies from (e.g. 'chrome:Profile 1', firefox).")
@click.option("--whisper-model", default=None, help="Whisper model (default: large-v3-turbo).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def ingest_youtube_cmd(url, chunk_duration, overlap, smart, extract, cookies_from, whisper_model, verbose):
    """Ingest a YouTube video by URL."""
    from .ingest import ingest_youtube
    from .store import VideoStore

    store = VideoStore()

    def progress(msg):
        click.echo(msg)

    try:
        result = ingest_youtube(
            url, store,
            chunk_duration=chunk_duration,
            overlap=overlap,
            smart_chunks=smart,
            whisper_model=whisper_model,
            extract_entities=extract,
            cookies_from_browser=cookies_from,
            verbose=verbose,
            on_progress=progress,
        )
        yt_note = ""
        if result.get("has_youtube_captions"):
            yt_note = " (YouTube captions + Whisper)"
        else:
            yt_note = " (Whisper only, no YouTube captions found)"

        click.echo(
            f"\nIngested \"{result['title']}\": {result['chunks']} chunks, "
            f"{result['transcribed']} transcribed{yt_note}."
        )
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)
    finally:
        store.close()


@ingest.command("instagram")
@click.argument("url")
@click.option("--extract/--no-extract", default=True, show_default=True,
              help="Extract entities to Obsidian vault after transcription.")
@click.option("--whisper-model", default=None, help="Whisper model (default: large-v3-turbo).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def ingest_instagram_cmd(url, extract, whisper_model, verbose):
    """Ingest an Instagram reel by URL."""
    from .ingest import ingest_instagram
    from .store import VideoStore

    store = VideoStore()

    def progress(msg):
        click.echo(msg)

    try:
        result = ingest_instagram(
            url, store,
            whisper_model=whisper_model,
            extract_entities=extract,
            verbose=verbose,
            on_progress=progress,
        )
        click.echo(
            f"\nIngested \"{result['title']}\": {result['chunks']} chunks, "
            f"{result['transcribed']} transcribed."
        )
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)
    finally:
        store.close()


# -----------------------------------------------------------------------
# search
# -----------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("-n", "--results", "n_results", default=5, show_default=True, help="Number of results.")
@click.option("-o", "--output-dir", default=".", show_default=True, help="Directory for trimmed clips.")
@click.option("--trim/--no-trim", default=True, show_default=True, help="Auto-trim top result.")
@click.option("--visual/--no-visual", default=False, show_default=True,
              help="Enable Tier 2 visual search (requires Gemini API key, costs money).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def search(query, n_results, output_dir, trim, visual, verbose):
    """Search indexed videos with a natural language QUERY.

    Searches transcripts first (free). Use --visual to enable paid Gemini
    embedding search when transcripts aren't enough.
    """
    from .search import search as do_search
    from .store import VideoStore

    store = VideoStore()
    stats = store.get_stats()

    if stats["total_chunks"] == 0:
        click.echo("No videos indexed yet. Run `vs ingest local <path>` first.")
        store.close()
        return

    results = do_search(
        query, store,
        limit=n_results,
        auto_embed=visual,
        verbose=verbose,
    )

    if not results:
        click.echo("No results found.")
        store.close()
        return

    # Check if we got an embed prompt
    embed_available = results[0].get("_embed_available", False) if results else False

    for i, r in enumerate(results, 1):
        basename = os.path.basename(r["source_file"])
        title = r.get("title") or basename
        start_str = _fmt_time(r["start_time"])
        end_str = _fmt_time(r["end_time"])
        score = r["score"]
        tier = r.get("search_tier", "?")
        source_type = r.get("source_type", "local")

        # Source badge
        badge = {"youtube": "YT", "instagram": "IG", "local": ""}.get(source_type, "")
        if badge:
            badge = f" [{badge}]"

        click.echo(f"  #{i} [{score:.3f}] T{tier}{badge} {title} @ {start_str}-{end_str}")

        # Show transcript snippet
        if r.get("transcript") and verbose:
            snippet = r["transcript"][:120].replace("\n", " ")
            click.echo(f"       {snippet}...")

    if embed_available and not visual:
        count = results[0].get("_embed_count", 0)
        cost = results[0].get("_embed_cost", 0)
        click.echo(
            f"\n  Transcript search only. {count} chunks not yet visually indexed."
            f"\n  Run with --visual to embed them (~${cost:.2f}) for visual/audio search."
        )

    if trim and results:
        top = results[0]
        if top.get("source_file") and os.path.isfile(top["source_file"]):
            from .trimmer import trim_clip, safe_filename
            output_path = os.path.join(
                output_dir,
                safe_filename(top["source_file"], top["start_time"], top["end_time"]),
            )
            try:
                clip_path = trim_clip(
                    top["source_file"], top["start_time"], top["end_time"], output_path,
                )
                click.echo(f"\nSaved clip: {clip_path}")
                _open_file(clip_path)
            except Exception as e:
                click.secho(f"\nCould not trim clip: {e}", fg="yellow", err=True)

    store.close()


# -----------------------------------------------------------------------
# cookies
# -----------------------------------------------------------------------

@cli.command("refresh-cookies")
@click.argument("browser", default="chrome:Profile 1")
def refresh_cookies_cmd(browser):
    """Export browser cookies for YouTube authentication.

    Saves cookies to ~/.videosearch/youtube-cookies.txt so yt-dlp can
    bypass bot detection without triggering macOS Keychain popups on
    every download. Re-run when cookies expire (yt-dlp will error with
    "cookies are no longer valid").

    BROWSER is the browser spec for yt-dlp (default: "chrome:Profile 1").
    Examples: "chrome", "chrome:Profile 1", "firefox", "safari".
    """
    import subprocess
    from pathlib import Path

    cookie_jar = Path.home() / ".videosearch" / "youtube-cookies.txt"
    cookie_jar.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Exporting cookies from {browser}...")
    click.echo("(You may see a macOS Keychain prompt -- allow access once.)")

    result = subprocess.run(
        [
            "yt-dlp",
            "--cookies-from-browser", browser,
            "--cookies", str(cookie_jar),
            "--dump-json", "--no-download",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        click.secho(f"Failed: {result.stderr.strip()}", fg="red", err=True)
        raise SystemExit(1)

    lines = sum(1 for _ in open(cookie_jar))
    click.echo(f"Saved {lines} cookies to {cookie_jar}")
    click.echo("Future YouTube ingests will use this file automatically.")


# -----------------------------------------------------------------------
# stats
# -----------------------------------------------------------------------

@cli.command()
def stats():
    """Show index statistics."""
    from .store import VideoStore

    store = VideoStore()
    s = store.get_stats()

    if s["total_chunks"] == 0:
        click.echo("Index is empty. Run `vs ingest` to add videos.")
        store.close()
        return

    click.echo(f"Videos:             {s['total_videos']}")
    click.echo(f"Chunks:             {s['total_chunks']}")
    click.echo(f"Transcribed:        {s['transcribed_chunks']}")
    click.echo(f"Visually embedded:  {s['embedded_chunks']}")

    if s["embedded_chunks"] < s["total_chunks"]:
        from .embedder import estimate_embedding_cost
        remaining = s["total_chunks"] - s["embedded_chunks"]
        cost = estimate_embedding_cost(remaining)
        click.echo(f"\n{remaining} chunks not yet visually indexed (~${cost:.2f} to embed).")

    store.close()


# -----------------------------------------------------------------------
# backfill
# -----------------------------------------------------------------------

@cli.command()
@click.option("--verbose", is_flag=True, help="Show debug info.")
def backfill(verbose):
    """Generate transcript notes in the vault for all indexed videos.

    Creates transcript markdown files with chunk headings for Obsidian
    deep-linking. Also updates video notes with transcript links.
    Skips videos that already have transcript notes.
    """
    from pathlib import Path
    from .extract import (
        _slugify, _create_transcript_note, _add_frontmatter_field,
        VAULT_PATH, TRANSCRIPTS_FOLDER, VIDEOS_FOLDER,
    )
    from .store import VideoStore

    store = VideoStore()
    videos = store.get_all_videos()

    if not videos:
        click.echo("No videos indexed. Run `vs ingest` first.")
        store.close()
        return

    created = 0
    skipped = 0

    for video in videos:
        slug = _slugify(video["title"] or "untitled")
        transcript_path = TRANSCRIPTS_FOLDER / f"{slug}.md"

        if transcript_path.exists():
            if verbose:
                click.echo(f"  Skip {video['title']} (transcript exists)")
            skipped += 1
            continue

        chunks = store.get_video_chunks(video["id"])
        if not chunks:
            skipped += 1
            continue

        # Build transcript list matching extract_and_persist format
        transcripts = []
        sources = set()
        for c in chunks:
            transcripts.append({
                "transcript": c["transcript"],
                "transcript_timestamped": c["transcript_timestamped"],
                "start_time": c["start_time"],
                "end_time": c["end_time"],
            })
            if c.get("transcript_source"):
                sources.add(c["transcript_source"])

        # Determine primary source
        if "whisper+youtube" in sources:
            source = "whisper+youtube"
        elif "youtube_captions" in sources and "whisper" in sources:
            source = "whisper+youtube"
        elif "youtube_captions" in sources:
            source = "youtube_captions"
        else:
            source = "whisper"

        path = _create_transcript_note(
            video["title"] or "untitled",
            video["source_type"],
            video.get("duration"),
            transcripts,
            transcript_source=source,
        )
        click.echo(f"  + {path.relative_to(VAULT_PATH)}")
        created += 1

        # Update video note with transcript link if missing
        video_note = VIDEOS_FOLDER / f"{slug}.md"
        if video_note.exists():
            content = video_note.read_text()
            if "transcript:" not in content:
                _add_frontmatter_field(video_note, "transcript", f'"[[{slug}]]"')
                if verbose:
                    click.echo(f"    Updated video note with transcript link")

    click.echo(f"\nBackfill complete: {created} transcripts created, {skipped} skipped.")
    store.close()
