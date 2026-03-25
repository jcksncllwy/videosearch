"""Entity extraction from video transcripts.

Extracts people, organizations, tools, and other entities from transcripts
and creates/updates structured notes in the Obsidian vault. Uses Claude
Sonnet via `claude -p` for structured extraction (Max quota, no API key).

The extraction follows the vault graph schema and entity-extract skill
conventions. Each entity becomes a markdown note with YAML frontmatter.
"""

import json
import os
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

VAULT_PATH = Path(os.environ.get("OBSIDIAN_VAULT", os.path.expanduser("~/obsidian/brain")))
VIDEOS_FOLDER = VAULT_PATH / "references" / "videos"
CONTACTS_FOLDER = VAULT_PATH / "references" / "contacts"
TOOLS_FOLDER = VAULT_PATH / "references" / "tools"

# Entity types we extract from video transcripts and their vault folders
ENTITY_FOLDERS = {
    "person": CONTACTS_FOLDER,
    "organization": CONTACTS_FOLDER,
    "tool": TOOLS_FOLDER,
    "video": VIDEOS_FOLDER,
}

EXTRACTION_PROMPT = """\
You are extracting structured entities from a video transcript for a knowledge graph.

VIDEO METADATA:
Title: {title}
Source: {source_type}
URL: {url}

TRANSCRIPT (with absolute timestamps):
{transcript}

Extract entities that are substantively discussed or referenced -- not just name-dropped in passing. For each entity, determine:
- name: proper name
- type: one of "person", "organization", "tool" (software/service/framework)
- context: one sentence about how they're mentioned in this video
- timestamp: the approximate timestamp where they're first mentioned (HH:MM:SS format, from the transcript timestamps)
- relationships: any relationships to other entities mentioned (e.g., "works for Datadog", "created by Google")

Return ONLY valid JSON in this exact format, no other text:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "person|organization|tool",
      "context": "Brief context of how they're mentioned",
      "timestamp": "00:01:35",
      "relationships": [
        {{"predicate": "worksFor|createdBy|uses", "target": "Other Entity Name"}}
      ]
    }}
  ]
}}

Rules:
- Only include entities mentioned substantively (discussed, explained, demonstrated), not passing references
- For people, include full names when available. Skip generic references ("someone", "a guy")
- For tools, include the specific tool/library/framework name, not generic categories
- If no meaningful entities are found, return {{"entities": []}}
- Timestamps should come from the transcript's timestamp markers
"""


def _slugify(name: str) -> str:
    """Convert a name to a kebab-case filename slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[''']s\b", "s", slug)  # possessives
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:80]


def _find_claude() -> str | None:
    """Find the claude CLI binary."""
    return shutil.which("claude")


def _call_sonnet(prompt: str) -> str | None:
    """Call Claude Sonnet via claude -p for structured extraction.

    Uses lean context flags to minimize token overhead. Default claude -p
    loads ~37K tokens (CLAUDE.md, all tools, all skills). Entity extraction
    is pure text-in/JSON-out -- no tools, no vault conventions, no skills
    needed. Lean flags cut context to ~12K tokens.
    """
    claude = _find_claude()
    if not claude:
        return None

    # Lean system prompt replaces full CLAUDE.md (~361 lines)
    system = (
        "You are an entity extraction assistant. "
        "Return ONLY valid JSON. No tools, no file access, no conversation."
    )

    try:
        env = os.environ.copy()
        env["ENABLE_TOOL_SEARCH"] = "false"

        result = subprocess.run(
            [
                claude, "-p", prompt,
                "--model", "sonnet",
                "--output-format", "text",
                "--max-turns", "1",
                "--system-prompt", system,
                "--disable-slash-commands",
            ],
            capture_output=True, text=True, timeout=120,
            env=env,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def _parse_extraction_response(response: str) -> list[dict]:
    """Parse JSON from Sonnet's response, handling markdown code fences."""
    if not response:
        return []

    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        return data.get("entities", [])
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r'\{[\s\S]*"entities"[\s\S]*\}', text)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("entities", [])
            except json.JSONDecodeError:
                pass
        return []


def _entity_exists(slug: str, entity_type: str) -> Path | None:
    """Check if an entity note already exists. Returns path if found."""
    folder = ENTITY_FOLDERS.get(entity_type)
    if not folder:
        return None

    candidate = folder / f"{slug}.md"
    if candidate.exists():
        return candidate

    # Also grep for the name across the vault for dedup
    return None


def _create_video_note(
    title: str,
    source_type: str,
    url: str | None,
    duration: float | None,
    entities: list[dict],
) -> Path:
    """Create a vault note for the video itself."""
    slug = _slugify(title)
    path = VIDEOS_FOLDER / f"{slug}.md"

    if path.exists():
        return path

    VIDEOS_FOLDER.mkdir(parents=True, exist_ok=True)

    tags = ["video", source_type]
    frontmatter = {
        "title": title,
        "type": "video",
        "tags": tags,
        "sourceType": source_type,
        "created": str(date.today()),
    }
    if url:
        frontmatter["url"] = url
    if duration:
        frontmatter["duration"] = round(duration)

    body_lines = [f"# {title}", ""]
    if url:
        body_lines.append(f"Source: {url}")
        body_lines.append("")

    if entities:
        body_lines.append("## Entities Mentioned")
        body_lines.append("")
        for e in entities:
            entity_slug = _slugify(e["name"])
            ts = e.get("timestamp", "")
            context = e.get("context", "")
            ts_str = f" at {ts}" if ts else ""
            body_lines.append(f"- [[{entity_slug}]]{ts_str} -- {context}")
        body_lines.append("")

    content = _format_note(frontmatter, "\n".join(body_lines))
    path.write_text(content)
    return path


def _create_or_update_entity(
    entity: dict,
    video_slug: str,
    video_title: str,
) -> tuple[Path | None, str]:
    """Create or update a vault note for an extracted entity.

    Returns (path, action) where action is 'created', 'updated', or 'skipped'.
    """
    name = entity.get("name", "").strip()
    entity_type = entity.get("type", "").lower()

    if not name or entity_type not in ENTITY_FOLDERS:
        return None, "skipped"

    slug = _slugify(name)
    folder = ENTITY_FOLDERS[entity_type]
    path = folder / f"{slug}.md"

    timestamp = entity.get("timestamp", "")
    context = entity.get("context", "")
    relationships = entity.get("relationships", [])

    ts_str = f" at {timestamp}" if timestamp else ""
    mention_line = f"Mentioned in [[{_slugify(video_title)}]]{ts_str} -- {context}"

    if path.exists():
        # Update: append mention if not already there
        existing = path.read_text()
        video_ref = _slugify(video_title)
        if video_ref not in existing:
            # Append to body
            updated = existing.rstrip() + f"\n\n{mention_line}\n"
            path.write_text(updated)

            # Add mentionedIn to frontmatter if not present
            if "mentionedIn:" not in existing:
                _add_frontmatter_field(path, "mentionedIn", f'"[[{video_ref}]]"')
            return path, "updated"
        return path, "skipped"

    # Create new note
    folder.mkdir(parents=True, exist_ok=True)

    if entity_type == "person":
        tags = ["contact", "person"]
        frontmatter = {
            "title": name,
            "type": "person",
            "tags": tags,
            "mentionedIn": f"[[{_slugify(video_title)}]]",
            "created": str(date.today()),
        }
        # Add relationships
        for rel in relationships:
            if rel.get("predicate") == "worksFor":
                frontmatter["worksFor"] = f"[[{_slugify(rel['target'])}]]"
    elif entity_type == "organization":
        tags = ["contact", "organization"]
        frontmatter = {
            "title": name,
            "type": "organization",
            "tags": tags,
            "mentionedIn": f"[[{_slugify(video_title)}]]",
            "created": str(date.today()),
        }
    elif entity_type == "tool":
        tags = ["tool"]
        frontmatter = {
            "title": name,
            "type": "tool",
            "tags": tags,
            "mentionedIn": f"[[{_slugify(video_title)}]]",
            "created": str(date.today()),
        }
        for rel in relationships:
            if rel.get("predicate") == "createdBy":
                frontmatter["createdBy"] = f"[[{_slugify(rel['target'])}]]"
    else:
        return None, "skipped"

    body = f"# {name}\n\n{mention_line}\n"
    content = _format_note(frontmatter, body)
    path.write_text(content)
    return path, "created"


def _format_note(frontmatter: dict, body: str) -> str:
    """Format a vault note with YAML frontmatter."""
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        elif isinstance(value, str) and value.startswith("[["):
            lines.append(f'{key}: "{value}"')
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


def _add_frontmatter_field(path: Path, key: str, value: str):
    """Add a field to existing YAML frontmatter."""
    content = path.read_text()
    # Find the closing --- and insert before it
    parts = content.split("---", 2)
    if len(parts) >= 3:
        fm = parts[1]
        if key not in fm:
            fm = fm.rstrip() + f"\n{key}: {value}\n"
            path.write_text(f"---{fm}---{parts[2]}")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_entities_from_transcript(
    transcript: str,
    title: str,
    source_type: str = "local",
    url: str | None = None,
) -> list[dict]:
    """Extract entities from a transcript using Claude Sonnet.

    Args:
        transcript: Full transcript text (may include timestamps).
        title: Video title.
        source_type: youtube, instagram, or local.
        url: Source URL if applicable.

    Returns:
        List of entity dicts from Sonnet's extraction.
    """
    prompt = EXTRACTION_PROMPT.format(
        title=title,
        source_type=source_type,
        url=url or "N/A",
        transcript=transcript[:8000],  # Cap to avoid token limits
    )

    response = _call_sonnet(prompt)
    return _parse_extraction_response(response)


def extract_and_persist(
    transcripts: list[dict],
    title: str,
    source_type: str = "local",
    url: str | None = None,
    duration: float | None = None,
    verbose: bool = False,
    on_progress=None,
) -> dict:
    """Extract entities from video transcripts and write to the vault.

    Args:
        transcripts: List of {transcript, transcript_timestamped, start_time, end_time}.
        title: Video title.
        source_type: youtube, instagram, local.
        url: Source URL.
        duration: Video duration in seconds.
        verbose: Print debug info.
        on_progress: Callback(message: str).

    Returns:
        Summary dict: {video_note, entities_created, entities_updated, entities_skipped}.
    """
    # Combine all timestamped transcripts for extraction
    # Prefer timestamped versions so Sonnet can report timestamps
    combined = []
    for t in transcripts:
        ts_text = t.get("transcript_timestamped") or t.get("transcript")
        if ts_text:
            combined.append(ts_text)

    full_transcript = "\n\n".join(combined)

    if not full_transcript.strip():
        if on_progress:
            on_progress("  No transcript content to extract from.")
        return {"video_note": None, "entities_created": 0, "entities_updated": 0, "entities_skipped": 0}

    if on_progress:
        on_progress("  Extracting entities via Sonnet...")

    entities = extract_entities_from_transcript(
        full_transcript, title, source_type, url,
    )

    if verbose and on_progress:
        on_progress(f"  Found {len(entities)} entities")

    # Create video note
    video_slug = _slugify(title)
    video_path = _create_video_note(title, source_type, url, duration, entities)
    if on_progress:
        on_progress(f"  Video note: {video_path.relative_to(VAULT_PATH)}")

    # Create/update entity notes
    created = 0
    updated = 0
    skipped = 0

    for entity in entities:
        path, action = _create_or_update_entity(entity, video_slug, title)
        if action == "created":
            created += 1
            if on_progress:
                on_progress(f"    + {entity['type']}: {entity['name']} ({path.name})")
        elif action == "updated":
            updated += 1
            if on_progress:
                on_progress(f"    ~ {entity['type']}: {entity['name']} (updated)")
        else:
            skipped += 1
            if verbose and on_progress:
                on_progress(f"    - {entity['name']} (skipped)")

    return {
        "video_note": str(video_path),
        "entities_created": created,
        "entities_updated": updated,
        "entities_skipped": skipped,
    }
