"""Tests for the SQLite store.

Rick Astley provides the test data. We index him so you don't have to.
"""

import math


class TestVideoStore:
    def test_add_video(self, store):
        vid = store.add_video(
            path="/tmp/rick.mp4",
            source_type="youtube",
            source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Never Gonna Give You Up",
            duration=212.0,
        )
        assert vid is not None
        assert len(vid) == 16

    def test_add_video_idempotent(self, store):
        """Adding the same video twice should not create duplicates."""
        vid1 = store.add_video(path="/tmp/rick.mp4", title="Rick")
        vid2 = store.add_video(path="/tmp/rick.mp4", title="Rick")
        assert vid1 == vid2
        assert store.get_stats()["total_videos"] == 1

    def test_is_video_indexed(self, store):
        assert not store.is_video_indexed("/tmp/rick.mp4")
        vid = store.add_video(path="/tmp/rick.mp4", title="Rick")
        # No chunks yet
        assert not store.is_video_indexed("/tmp/rick.mp4")
        # Add a chunk
        store.add_chunk(
            video_id=vid, start_time=0.0, end_time=30.0,
            transcript="We're no strangers to love",
        )
        assert store.is_video_indexed("/tmp/rick.mp4")

    def test_add_chunk_with_transcript(self, store):
        vid = store.add_video(path="/tmp/rick.mp4", title="Rick")
        chunk_id = store.add_chunk(
            video_id=vid, start_time=0.0, end_time=30.0,
            transcript="We're no strangers to love, you know the rules and so do I",
            transcript_source="whisper",
        )
        assert chunk_id is not None
        stats = store.get_stats()
        assert stats["total_chunks"] == 1
        assert stats["transcribed_chunks"] == 1
        assert stats["embedded_chunks"] == 0

    def test_search_transcripts(self, store):
        vid = store.add_video(path="/tmp/rick.mp4", title="Never Gonna Give You Up")
        store.add_chunk(
            video_id=vid, start_time=0.0, end_time=30.0,
            transcript="We're no strangers to love, you know the rules and so do I",
        )
        store.add_chunk(
            video_id=vid, start_time=25.0, end_time=55.0,
            transcript="A full commitment's what I'm thinking of, you wouldn't get this from any other guy",
        )
        store.add_chunk(
            video_id=vid, start_time=50.0, end_time=80.0,
            transcript="Never gonna give you up, never gonna let you down",
        )

        results = store.search_transcripts("strangers to love")
        assert len(results) >= 1
        assert results[0]["start_time"] == 0.0
        assert results[0]["search_tier"] == 1
        assert "Never Gonna Give You Up" in results[0]["title"]

    def test_search_transcripts_no_results(self, store):
        vid = store.add_video(path="/tmp/rick.mp4", title="Rick")
        store.add_chunk(
            video_id=vid, start_time=0.0, end_time=30.0,
            transcript="Never gonna give you up",
        )
        results = store.search_transcripts("dancing trench coat")
        assert len(results) == 0

    def test_embedding_storage_and_search(self, store):
        vid = store.add_video(path="/tmp/rick.mp4", title="Rick")
        # Create a fake 768-dim embedding
        embedding = [0.1] * 768
        chunk_id = store.add_chunk(
            video_id=vid, start_time=0.0, end_time=30.0,
            transcript="Never gonna give you up",
            embedding=embedding,
        )
        stats = store.get_stats()
        assert stats["embedded_chunks"] == 1

        # Search with a similar embedding
        query_emb = [0.1] * 768
        results = store.search_embeddings(query_emb, limit=5)
        assert len(results) == 1
        assert results[0]["score"] > 0.99  # Near-identical vectors
        assert results[0]["search_tier"] == 2

    def test_lazy_embedding(self, store):
        """Embeddings can be added after initial chunk creation."""
        vid = store.add_video(path="/tmp/rick.mp4", title="Rick")
        chunk_id = store.add_chunk(
            video_id=vid, start_time=0.0, end_time=30.0,
            transcript="Never gonna give you up",
        )

        # Initially no embeddings
        assert store.get_stats()["embedded_chunks"] == 0
        unembedded = store.get_unembedded_chunks()
        assert len(unembedded) == 1

        # Add embedding lazily
        store.set_embedding(chunk_id, [0.5] * 768)
        assert store.get_stats()["embedded_chunks"] == 1
        assert len(store.get_unembedded_chunks()) == 0

    def test_stats_empty(self, store):
        stats = store.get_stats()
        assert stats["total_videos"] == 0
        assert stats["total_chunks"] == 0


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from glean.store import _cosine_similarity
        a = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, a) == 1.0

    def test_orthogonal_vectors(self):
        from glean.store import _cosine_similarity
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-10

    def test_opposite_vectors(self):
        from glean.store import _cosine_similarity
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == -1.0
