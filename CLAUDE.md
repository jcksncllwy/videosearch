# VideoSearch

Personal video search engine. Ingest videos from local files, YouTube, or Instagram. Search with natural language using a cost cascade: free transcript search first, paid Gemini visual search only when needed.

## Stack

- Python 3.10+, Click CLI
- SQLite with FTS5 for transcript search
- whisper.cpp (via ~/ygg/bin/transcribe) for local transcription
- Gemini Embedding 2 for visual/audio embedding (Tier 2, paid, lazy)
- yt-dlp for YouTube/Instagram downloads
- ffmpeg for chunking and trimming

## Architecture

### Cost Cascade

1. **Ingest** always runs free processing: chunk video, extract audio, transcribe with Whisper
2. **Search Tier 1**: FTS5 full-text search over transcripts (free, instant)
3. **Search Tier 2**: Gemini embedding similarity (paid, ~$2.84/hr of video, lazy -- only embeds chunks on first visual search)

### Data Model

- `~/.videosearch/videosearch.db` -- SQLite database
- Videos table: metadata, source type, URLs
- Chunks table: transcripts, timestamps, embeddings (nullable)
- FTS5 virtual table for transcript search

## CLI

```
vs ingest local <path>          # Ingest local video files
vs ingest youtube <url>         # Download and ingest YouTube video
vs ingest instagram <url>       # Download and ingest Instagram reel
vs search "query"               # Search (transcript first)
vs search "query" --visual      # Search with Tier 2 visual search
vs stats                        # Show index stats
```

## Dependencies

- ffmpeg (system)
- whisper-cli / whisper.cpp (system, via brew)
- yt-dlp (system, via brew)
- GEMINI_API_KEY in .env (only for Tier 2)
