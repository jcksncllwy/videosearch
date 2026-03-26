"""Gemini embedding client for Tier 2 visual search.

Only used when transcript search (Tier 1) can't find what you need.
Requires GEMINI_API_KEY env var.
"""

import os
import sys
import time
from collections import deque

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "gemini-embedding-2"
DIMENSIONS = 768
DEFAULT_RPM = 55


class _RateLimiter:
    """Sliding-window rate limiter."""

    def __init__(self, max_per_minute: int = DEFAULT_RPM):
        self._max = max_per_minute
        self._timestamps: deque[float] = deque()

    def wait(self):
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            sleep_for = 60.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


_limiter = _RateLimiter()
_client = None


class GeminiKeyError(RuntimeError):
    pass


class GeminiQuotaError(RuntimeError):
    pass


def _get_client():
    global _client
    if _client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GeminiKeyError(
                "GEMINI_API_KEY not set. Get a key at https://aistudio.google.com/apikey\n"
                "Add it to .env or export it. Only needed for Tier 2 visual search."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _retry(fn, max_retries: int = 5, initial_delay: float = 2.0):
    """Exponential backoff on transient errors."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            retryable = status in (429, 503) or "resource exhausted" in msg or "503" in msg
            if not retryable or attempt == max_retries:
                if "resource exhausted" in msg or status == 429:
                    raise GeminiQuotaError(
                        "Gemini rate limit hit. Wait a minute or upgrade your plan."
                    ) from exc
                raise
            wait = min(delay, 60.0)
            print(f"  Retrying ({attempt + 1}/{max_retries}), waiting {wait:.0f}s...", file=sys.stderr)
            time.sleep(wait)
            delay *= 2


def embed_video_chunk(chunk_path: str, verbose: bool = False) -> list[float]:
    """Embed a video chunk (with audio) via Gemini. Returns 768-dim vector."""
    from google.genai import types

    client = _get_client()
    with open(chunk_path, "rb") as f:
        video_bytes = f.read()

    if hasattr(types.Part, "from_bytes"):
        video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    else:
        video_part = types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"))

    if verbose:
        size_kb = len(video_bytes) / 1024
        print(f"    [embed] sending {size_kb:.0f}KB to {EMBED_MODEL}", file=sys.stderr)

    _limiter.wait()
    t0 = time.monotonic()
    response = _retry(
        lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=types.Content(parts=[video_part]),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=DIMENSIONS,
            ),
        )
    )
    elapsed = time.monotonic() - t0
    embedding = response.embeddings[0].values

    if verbose:
        print(f"    [embed] dims={len(embedding)}, time={elapsed:.2f}s", file=sys.stderr)

    return embedding


def embed_query(query_text: str, verbose: bool = False) -> list[float]:
    """Embed a text query for search."""
    from google.genai import types

    client = _get_client()
    _limiter.wait()
    t0 = time.monotonic()
    response = _retry(
        lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=query_text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=DIMENSIONS,
            ),
        )
    )
    elapsed = time.monotonic() - t0
    embedding = response.embeddings[0].values

    if verbose:
        print(f"  [embed] query dims={len(embedding)}, time={elapsed:.2f}s", file=sys.stderr)

    return embedding


def estimate_embedding_cost(n_chunks: int, chunk_duration: int = 30) -> float:
    """Estimate Gemini API cost for embedding N chunks.

    Based on $0.00079/frame, 1 frame/sec extracted by the API.
    """
    frames_per_chunk = chunk_duration  # 1 fps
    return n_chunks * frames_per_chunk * 0.00079
