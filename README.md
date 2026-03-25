# VideoSearch

Search your video collection with natural language. Ingest videos from local files, YouTube, or Instagram. Search transcripts instantly for free -- only pay for visual search when you need it.

## How it works

1. **Ingest** -- download (if needed), chunk at natural audio breaks, transcribe with Whisper. All free, all local.
2. **Search** -- full-text search over transcripts. If that's not enough, optionally run visual search via Gemini embeddings (paid, cached).

```
$ vs ingest youtube "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
Downloading: Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)
  Found YouTube captions (61 entries)
  Chunk 1/8: transcribing...
  ...
  Chunk 8/8: transcribing...

Ingested "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)":
  8 chunks, 8 transcribed (YouTube captions + Whisper).

$ vs search "known each other for so long" --no-trim
  #1 [6.555] T1 [YT] Rick Astley - Never Gonna Give You Up @ 02:00-02:30
  #2 [5.740] T1 [YT] Rick Astley - Never Gonna Give You Up @ 01:00-01:30
```

## Install

```bash
git clone https://github.com/jcksncllwy/videosearch.git
cd videosearch
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

System dependencies:

```bash
brew install ffmpeg whisper-cpp yt-dlp
```

## Usage

### Ingest

```bash
# Local video files (file or directory)
vs ingest local ~/Videos/meetings/

# YouTube
vs ingest youtube "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Instagram reels
vs ingest instagram "https://www.instagram.com/reel/ABC123/"
```

Ingestion is free. Videos are chunked at natural silence breaks (smart chunking), then transcribed locally with Whisper. YouTube captions are grabbed too when available.

### Search

```bash
# Transcript search (free, instant)
vs search "never gonna give you up"

# With visual search (requires Gemini API key, costs money)
vs search "man dancing in a trench coat" --visual
```

Search uses a cost cascade:

| Tier | Method | Cost | Good for |
|------|--------|------|----------|
| 1 | Transcript FTS | Free | Dialogue, lyrics, spoken words |
| 2 | Gemini embedding | ~$2.84/hr of video | Visual content, sounds, vibes |

Tier 2 embeddings are cached -- you only pay once per chunk.

### Options

```bash
# Chunking
--chunk-duration 30   # Target seconds per chunk (default: 30)
--no-smart            # Disable smart chunking (fixed intervals instead)

# Search
--no-trim             # Don't auto-extract a clip
--visual              # Enable Tier 2 visual search
-n 10                 # Number of results
-o ~/clips/           # Output directory for trimmed clips

# Debug
--verbose             # Show chunking decisions, API timing, etc.
```

### Stats

```bash
$ vs stats
Videos:             1
Chunks:             8
Transcribed:        8
Visually embedded:  0

8 chunks not yet visually indexed (~$0.21 to embed).
```

## Smart chunking

By default, VideoSearch detects silence gaps in the audio and snaps chunk boundaries to natural pauses. This prevents sentences and points of interest from being split across chunks. Falls back to fixed intervals when no silence is detected (e.g., continuous music).

```
$ vs ingest youtube "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --verbose
  [chunk] Detected 3 silence gaps
  [chunk] No silence near 30.0s, using fixed boundary
  ...
  [chunk] Snapped 210.0s -> 209.7s (silence)
```

Rick Astley has almost no silence (the man does not stop), so most boundaries stay fixed. A podcast or meeting recording will show much more snapping.

## Absolute timestamps

Every transcript line includes timestamps that map directly to the source video:

```
[00:01:00.000 --> 00:01:04.400]   We've known each other for so long
[00:01:04.400 --> 00:01:09.200]   Your heart's been aching but you're too shy to say it
```

These are absolute -- `01:00` means one minute into the original video, not one minute into the chunk.

## Gemini visual search (Tier 2)

For queries that transcripts can't answer ("the slide with the red chart", "when the dog runs through the background"), enable visual search:

```bash
# Set up your API key (one time)
echo 'GEMINI_API_KEY=your-key' > .env

# Search with visual
vs search "man dancing in a trench coat" --visual
```

Visual search uses Gemini Embedding 2, which embeds video and audio natively -- no frame captioning or transcription middleman. Embeddings are computed lazily (only when you search with `--visual`) and cached permanently.

## Requirements

- Python 3.10+
- ffmpeg
- whisper-cli (whisper.cpp, `brew install whisper-cpp`)
- yt-dlp (`brew install yt-dlp`)
- Gemini API key (optional, only for Tier 2 visual search)
