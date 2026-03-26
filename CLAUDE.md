# VideoSearch

Personal video search engine. Ingest videos from local files, YouTube, or Instagram. Search with natural language using a cost cascade: free transcript search first, paid Gemini visual search only when needed. Automatically extracts entities to the Obsidian vault knowledge graph.

## Stack

- Python 3.10+, Click CLI (`vs`)
- SQLite with FTS5 for transcript search
- whisper.cpp (via `~/ygg/bin/transcribe`) for local transcription
- Gemini Embedding 2 for visual/audio embedding (Tier 2, paid, lazy)
- yt-dlp for YouTube/Instagram downloads
- ffmpeg for chunking and trimming
- Claude Sonnet via `claude -p` for entity extraction (Max quota, no API key)

## Architecture

### Cost Cascade

1. **Ingest** always runs free processing: download (if needed) -> chunk video -> extract audio -> transcribe with Whisper -> extract entities to vault
2. **Search Tier 1**: FTS5 full-text search over transcripts (free, instant)
3. **Search Tier 2**: Gemini embedding similarity (paid, ~$2.84/hr of video, lazy -- only embeds chunks on first visual search)

### Data Model

- `~/.videosearch/videosearch.db` -- SQLite database
- `videos` table: metadata, source type, URLs, description, channel
- `chunks` table: transcripts (plain + timestamped), timestamps, embeddings (nullable), transcript source
- `chunks_fts` FTS5 virtual table for transcript search

### Entity Extraction Pipeline

During ingest, entities are extracted from transcripts and written to the Obsidian vault:

1. Combined timestamped transcript + video description + channel name sent to Sonnet
2. Sonnet returns structured JSON: entities with types, timestamps, relationships
3. For each entity, a vault note is created or updated with:
   - YAML frontmatter per entity schema (person, organization, tool, account)
   - Frontmatter relationship backfill (worksFor, createdBy, managedBy) -- adds missing fields, never overwrites
   - Rich mention blocks in a `## Video Mentions` section with transcript deep-links
4. A transcript note is created with H2 chunk headings (`## 00:05:30 - 00:06:00`) enabling Obsidian `#heading` deep-links
5. A video note is created linking to the transcript and listing mentioned entities
6. The publishing channel gets an Account entity note

**Lean context for Sonnet**: Entity extraction uses `claude -p` with `--system-prompt`, `--disable-slash-commands`, and `ENABLE_TOOL_SEARCH=false` to cut context from ~37K to ~12K tokens. Pure text-in/JSON-out -- no tools needed.

**Hallucination guard**: System prompt includes grounding constraint: "Only extract entities that are explicitly named in the provided transcript or description text."

### Entity Types

| Type | Vault folder | Schema doc |
|------|-------------|------------|
| person | `references/contacts/` | `entity-schemas/person.md` |
| organization | `references/contacts/` | `entity-schemas/organization.md` |
| tool | `references/tools/` | `entity-schemas/tool.md` |
| account | `references/accounts/` | `entity-schemas/account.md` |
| video | `references/videos/` | `entity-schemas/video.md` |
| transcript | `references/transcripts/` | `entity-schemas/transcript.md` |

### Transcript Deep-Linking

Transcript notes use H2 chunk headings like `## 00:05:30 - 00:06:00`. Entity notes reference specific moments via `[[transcript-slug#00:05:30 - 00:06:00|transcript]]`. This enables click-through from any entity to the exact transcript passage where it was mentioned.

## CLI

```
vs ingest local <path>          # Ingest local video files
vs ingest youtube <url>         # Download and ingest YouTube video
vs ingest instagram <url>       # Download and ingest Instagram reel
vs search "query"               # Search (transcript first)
vs search "query" --visual      # Search with Tier 2 visual search
vs stats                        # Show index stats
vs backfill                     # Generate transcript notes for all indexed videos
vs refresh-cookies [browser]    # Export browser cookies for YouTube auth
```

### Common Options

- `--extract/--no-extract` -- toggle entity extraction (default: on)
- `--smart/--no-smart` -- snap chunk boundaries to silence gaps (default: on)
- `--whisper-model MODEL` -- override Whisper model (default: large-v3-turbo)
- `--cookies-from BROWSER` -- browser spec for YouTube cookies (e.g. "chrome:Profile 1")
- `--verbose` -- show debug info

## YouTube Cookie Management

YouTube frequently blocks yt-dlp with "Sign in to confirm you're not a bot." The cookie jar pattern avoids this:

1. **Export once**: `vs refresh-cookies "chrome:Profile 1"` exports browser cookies to `~/.videosearch/youtube-cookies.txt`. This triggers one macOS Keychain popup.
2. **Auto-detected**: All subsequent `vs ingest youtube` calls automatically use the jar file -- no Keychain popups, no `--cookies-from` flag needed.
3. **Refresh when needed**: When cookies expire, yt-dlp will error. Run `vs refresh-cookies` again.

The cookie jar is checked first. `--cookies-from` is a fallback that extracts live from the browser (triggers Keychain popup each time).

## Module Map

| Module | Purpose |
|--------|---------|
| `cli.py` | Click CLI entry point, all commands |
| `ingest.py` | Download + chunking + transcription pipelines per source type |
| `store.py` | SQLite store with FTS5, embedding storage, schema migrations |
| `extract.py` | Entity extraction via Sonnet, vault note creation/update |
| `search.py` | Tier 1 (FTS) + Tier 2 (embedding) search orchestration |
| `chunker.py` | Video chunking with smart silence-boundary snapping |
| `transcribe.py` | Whisper transcription wrapper |
| `embedder.py` | Gemini embedding for Tier 2 visual search |
| `trimmer.py` | ffmpeg clip trimming for search results |

## Dependencies

### System (brew)

- `ffmpeg` -- video processing
- `whisper-cpp` -- local transcription (the `whisper-cli` formula)
- `yt-dlp` -- YouTube/Instagram downloading

### Python (pip)

- `click` -- CLI framework
- `python-dotenv` -- .env loading
- `google-genai` -- Gemini API (Tier 2 only)
- `ffmpeg-python`, `imageio-ffmpeg` -- ffmpeg bindings

### Environment

- `GEMINI_API_KEY` in `.env` -- only needed for Tier 2 visual search
- `OBSIDIAN_VAULT` -- vault path override (default: `~/obsidian/brain`)
- `claude` CLI -- must be on PATH for entity extraction

## Development

```bash
pip install -e ".[dev]"     # Install in dev mode
pytest                       # Run tests
pytest -m "not slow"         # Skip download/transcription tests
```
