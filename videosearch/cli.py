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
@click.option("--whisper-model", default=None, help="Whisper model (default: large-v3-turbo).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def ingest_local_cmd(path, chunk_duration, overlap, smart, whisper_model, verbose):
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
@click.option("--whisper-model", default=None, help="Whisper model (default: large-v3-turbo).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def ingest_youtube_cmd(url, chunk_duration, overlap, smart, whisper_model, verbose):
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
@click.option("--whisper-model", default=None, help="Whisper model (default: large-v3-turbo).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def ingest_instagram_cmd(url, whisper_model, verbose):
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
