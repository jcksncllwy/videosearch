"""SQLite store with FTS5 for transcript search and optional embedding storage.

Single-file database. No external vector DB dependency.
Embeddings stored as blobs, cosine similarity computed in Python.
"""

import hashlib
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".glean" / "glean.db"


def _make_chunk_id(source_file: str, start_time: float) -> str:
    raw = f"{source_file}:{start_time:.2f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VideoStore:
    """SQLite-backed store for video chunks, transcripts, and embeddings."""

    def __init__(self, db_path: str | Path | None = None):
        db_path = str(db_path or DEFAULT_DB_PATH)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        c = self._conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'local',
                source_url TEXT,
                title TEXT,
                description TEXT,
                channel TEXT,
                duration REAL,
                added_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES videos(id),
                chunk_path TEXT,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                transcript TEXT,
                transcript_timestamped TEXT,
                transcript_source TEXT,
                embedding BLOB,
                indexed_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);
        """)

        # Migrate: add description and channel columns if missing
        cols = {row[1] for row in c.execute("PRAGMA table_info(videos)").fetchall()}
        if "description" not in cols:
            c.execute("ALTER TABLE videos ADD COLUMN description TEXT")
        if "channel" not in cols:
            c.execute("ALTER TABLE videos ADD COLUMN channel TEXT")

        # FTS5 virtual table for transcript search
        # Check if it exists first (CREATE IF NOT EXISTS doesn't work for virtual tables)
        existing = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        if not existing:
            c.execute("""
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    transcript,
                    content='chunks',
                    content_rowid='rowid'
                )
            """)

        c.commit()

    # ------------------------------------------------------------------
    # Videos
    # ------------------------------------------------------------------

    def add_video(
        self,
        path: str,
        source_type: str = "local",
        source_url: str | None = None,
        title: str | None = None,
        description: str | None = None,
        channel: str | None = None,
        duration: float | None = None,
    ) -> str:
        """Register a video. Returns video ID."""
        video_id = hashlib.sha256(path.encode()).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR IGNORE INTO videos
               (id, path, source_type, source_url, title, description, channel, duration, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (video_id, path, source_type, source_url, title, description, channel, duration, now),
        )
        self._conn.commit()
        return video_id

    def is_video_indexed(self, path: str) -> bool:
        """Check if a video has any chunks."""
        video_id = hashlib.sha256(path.encode()).hexdigest()[:16]
        row = self._conn.execute(
            "SELECT 1 FROM chunks WHERE video_id = ? LIMIT 1", (video_id,)
        ).fetchone()
        return row is not None

    def video_has_embeddings(self, path: str) -> bool:
        """Check if a video's chunks have Gemini embeddings."""
        video_id = hashlib.sha256(path.encode()).hexdigest()[:16]
        row = self._conn.execute(
            "SELECT 1 FROM chunks WHERE video_id = ? AND embedding IS NOT NULL LIMIT 1",
            (video_id,),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
        chunk_path: str | None = None,
        transcript: str | None = None,
        transcript_timestamped: str | None = None,
        transcript_source: str | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Add a chunk with transcript and/or embedding."""
        video_row = self._conn.execute(
            "SELECT path FROM videos WHERE id = ?", (video_id,)
        ).fetchone()
        source_file = video_row["path"] if video_row else video_id
        chunk_id = _make_chunk_id(source_file, start_time)
        now = datetime.now(timezone.utc).isoformat()

        emb_blob = None
        if embedding:
            emb_blob = json.dumps(embedding).encode()

        # Clean up old FTS entry before replacing the chunk (INSERT OR REPLACE
        # deletes+reinserts, changing the rowid and orphaning the old FTS entry)
        old_row = self._conn.execute(
            "SELECT rowid FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if old_row:
            self._conn.execute(
                "DELETE FROM chunks_fts WHERE rowid = ?", (old_row[0],)
            )

        self._conn.execute(
            """INSERT OR REPLACE INTO chunks
               (id, video_id, chunk_path, start_time, end_time,
                transcript, transcript_timestamped, transcript_source,
                embedding, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, video_id, chunk_path, start_time, end_time,
             transcript, transcript_timestamped, transcript_source,
             emb_blob, now),
        )

        # Update FTS index
        if transcript:
            rowid = self._conn.execute(
                "SELECT rowid FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()[0]
            self._conn.execute(
                "INSERT INTO chunks_fts(rowid, transcript) VALUES (?, ?)",
                (rowid, transcript),
            )

        self._conn.commit()
        return chunk_id

    def set_embedding(self, chunk_id: str, embedding: list[float]):
        """Lazily add an embedding to an existing chunk."""
        emb_blob = json.dumps(embedding).encode()
        self._conn.execute(
            "UPDATE chunks SET embedding = ? WHERE id = ?",
            (emb_blob, chunk_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Search: Tier 1 (transcript FTS)
    # ------------------------------------------------------------------

    def search_transcripts(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search over transcripts. Free, fast."""
        rows = self._conn.execute(
            """SELECT c.id, c.video_id, c.start_time, c.end_time,
                      c.transcript, c.transcript_source, c.chunk_path,
                      v.path as source_file, v.title, v.description, v.channel,
                      v.source_type, v.source_url,
                      rank
               FROM chunks_fts fts
               JOIN chunks c ON c.rowid = fts.rowid
               JOIN videos v ON v.id = c.video_id
               WHERE chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()

        results = []
        for row in rows:
            results.append({
                "chunk_id": row["id"],
                "source_file": row["source_file"],
                "title": row["title"],
                "description": row["description"],
                "channel": row["channel"],
                "source_type": row["source_type"],
                "source_url": row["source_url"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "transcript": row["transcript"],
                "transcript_source": row["transcript_source"],
                "chunk_path": row["chunk_path"],
                "score": -row["rank"],  # FTS5 rank is negative (lower = better)
                "search_tier": 1,
            })
        return results

    # ------------------------------------------------------------------
    # Search: Tier 2 (embedding similarity)
    # ------------------------------------------------------------------

    def search_embeddings(
        self, query_embedding: list[float], limit: int = 10,
    ) -> list[dict]:
        """Cosine similarity search over stored embeddings. Requires Gemini."""
        rows = self._conn.execute(
            """SELECT c.id, c.video_id, c.start_time, c.end_time,
                      c.transcript, c.chunk_path, c.embedding,
                      v.path as source_file, v.title, v.source_type, v.source_url
               FROM chunks c
               JOIN videos v ON v.id = c.video_id
               WHERE c.embedding IS NOT NULL"""
        ).fetchall()

        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            sim = _cosine_similarity(query_embedding, emb)
            scored.append({
                "chunk_id": row["id"],
                "source_file": row["source_file"],
                "title": row["title"],
                "source_type": row["source_type"],
                "source_url": row["source_url"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "transcript": row["transcript"],
                "chunk_path": row["chunk_path"],
                "score": sim,
                "search_tier": 2,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def get_unembedded_chunks(self) -> list[dict]:
        """Return chunks that don't have embeddings yet (for lazy Tier 2)."""
        rows = self._conn.execute(
            """SELECT c.id, c.video_id, c.chunk_path, c.start_time, c.end_time,
                      v.path as source_file
               FROM chunks c
               JOIN videos v ON v.id = c.video_id
               WHERE c.embedding IS NULL"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def get_all_videos(self) -> list[dict]:
        """Return all videos with metadata."""
        rows = self._conn.execute(
            "SELECT id, path, source_type, source_url, title, description, channel, duration FROM videos"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_video_chunks(self, video_id: str) -> list[dict]:
        """Return all chunks for a video, ordered by start time."""
        rows = self._conn.execute(
            """SELECT id, start_time, end_time, transcript, transcript_timestamped,
                      transcript_source, chunk_path
               FROM chunks WHERE video_id = ? ORDER BY start_time""",
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_videos = self._conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        embedded_chunks = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        transcribed_chunks = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE transcript IS NOT NULL"
        ).fetchone()[0]

        return {
            "total_videos": total_videos,
            "total_chunks": total_chunks,
            "transcribed_chunks": transcribed_chunks,
            "embedded_chunks": embedded_chunks,
        }

    def close(self):
        self._conn.close()
