"""Search with cost cascade: free transcript search first, paid embeddings on demand."""

import click

from .store import VideoStore


def search(
    query: str,
    store: VideoStore,
    limit: int = 10,
    transcript_threshold: float = 0.0,
    embedding_threshold: float = 0.35,
    auto_embed: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Search with cost cascade.

    Tier 1: Transcript FTS (free, instant).
    Tier 2: Gemini embedding similarity (paid, lazy).

    Args:
        query: Natural language search string.
        store: VideoStore instance.
        limit: Max results.
        transcript_threshold: Minimum FTS score to consider a transcript hit "good enough".
        embedding_threshold: Minimum cosine similarity for embedding results.
        auto_embed: If True, skip confirmation and embed unembedded chunks automatically.
        verbose: Print debug info.

    Returns:
        List of result dicts, each with 'search_tier' indicating which tier found it.
    """
    # --- Tier 1: Transcript search (free) ---
    if verbose:
        click.echo("  [search] Tier 1: transcript FTS...", err=True)

    transcript_results = store.search_transcripts(query, limit=limit)

    if transcript_results and transcript_results[0]["score"] > transcript_threshold:
        if verbose:
            click.echo(
                f"  [search] Tier 1 hit: {len(transcript_results)} results, "
                f"best score={transcript_results[0]['score']:.4f}",
                err=True,
            )
        return transcript_results

    # --- Tier 2: Embedding search (paid) ---
    if verbose:
        click.echo("  [search] Tier 1 insufficient, checking Tier 2...", err=True)

    # Check if we have any embeddings at all
    stats = store.get_stats()
    unembedded = store.get_unembedded_chunks()

    if stats["embedded_chunks"] > 0:
        # Search what we have
        if verbose:
            click.echo(
                f"  [search] Searching {stats['embedded_chunks']} embedded chunks...",
                err=True,
            )
        from .embedder import embed_query
        query_embedding = embed_query(query, verbose=verbose)
        embedding_results = store.search_embeddings(query_embedding, limit=limit)

        if embedding_results and embedding_results[0]["score"] >= embedding_threshold:
            return embedding_results

    if unembedded:
        # There are chunks we haven't embedded yet
        from .embedder import estimate_embedding_cost
        cost = estimate_embedding_cost(len(unembedded))

        if not auto_embed:
            return _make_embed_prompt(
                transcript_results, unembedded, cost
            )

        # Auto-embed mode: go ahead and embed everything
        if verbose:
            click.echo(f"  [search] Embedding {len(unembedded)} chunks (~${cost:.2f})...", err=True)
        _embed_chunks(store, unembedded, verbose=verbose)

        from .embedder import embed_query
        query_embedding = embed_query(query, verbose=verbose)
        return store.search_embeddings(query_embedding, limit=limit)

    # Nothing found anywhere
    return transcript_results  # Return whatever Tier 1 had, even if weak


def _make_embed_prompt(
    transcript_results: list[dict],
    unembedded: list[dict],
    cost: float,
) -> list[dict]:
    """Return transcript results annotated with an embed suggestion."""
    # Mark results to signal the CLI that embedding is available
    for r in transcript_results:
        r["_embed_available"] = True
        r["_embed_count"] = len(unembedded)
        r["_embed_cost"] = cost
    return transcript_results


def _embed_chunks(
    store: VideoStore,
    chunks: list[dict],
    verbose: bool = False,
):
    """Embed a list of chunks and store the embeddings."""
    from .embedder import embed_video_chunk

    for i, chunk in enumerate(chunks):
        if verbose:
            click.echo(f"  [embed] chunk {i + 1}/{len(chunks)}", err=True)
        if not chunk["chunk_path"] or not __import__("os").path.isfile(chunk["chunk_path"]):
            continue
        embedding = embed_video_chunk(chunk["chunk_path"], verbose=verbose)
        store.set_embedding(chunk["id"], embedding)
