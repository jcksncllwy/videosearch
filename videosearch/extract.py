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
ACCOUNTS_FOLDER = VAULT_PATH / "references" / "accounts"

# Entity types we extract from video transcripts and their vault folders
ENTITY_FOLDERS = {
    "person": CONTACTS_FOLDER,
    "organization": CONTACTS_FOLDER,
    "tool": TOOLS_FOLDER,
    "account": ACCOUNTS_FOLDER,
    "video": VIDEOS_FOLDER,
}

# Known social platform URL patterns -> platform identifiers
PLATFORM_PATTERNS = [
    (r"youtube\.com|youtu\.be", "youtube"),
    (r"instagram\.com", "instagram"),
    (r"twitter\.com|x\.com", "x"),
    (r"twitch\.tv", "twitch"),
    (r"tiktok\.com", "tiktok"),
    (r"github\.com", "github"),
    (r"linkedin\.com", "linkedin"),
    (r"spotify\.com", "spotify"),
    (r"discord\.gg", "discord"),
]

EXTRACTION_PROMPT = """\
You are extracting structured entities from a video transcript for a knowledge graph.

VIDEO METADATA:
Title: {title}
Source: {source_type}
URL: {url}
Channel: {channel}
Description: {description}

TRANSCRIPT (with absolute timestamps):
{transcript}

Extract entities that are substantively discussed or referenced. For each entity, determine:
- name: proper name or display name
- type: one of "person", "organization", "tool" (software/service/framework), "account" (social media / web platform account)
- context: one sentence about how they're mentioned in this video
- timestamp: the approximate timestamp where they're first mentioned (HH:MM:SS format, from the transcript timestamps). Use "00:00:00" for entities found only in the description.
- relationships: any relationships to other entities mentioned (e.g., "worksFor: Datadog", "createdBy: Google", "managedBy: Ludwig Ahgren")

For accounts specifically:
- Extract social media accounts mentioned in the video or description (YouTube channels, Instagram, X/Twitter, Twitch, TikTok, etc.)
- Include the platform in the name: "Ludwig (YouTube)", "@ludwigahgren (Instagram)"
- Use "platform" field to indicate which platform
- Use "handle" field for the username/handle (without @)
- Use "managedBy" relationship to link to the person or org who runs it

Return ONLY valid JSON in this exact format, no other text:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "person|organization|tool|account",
      "context": "Brief context of how they're mentioned",
      "timestamp": "00:01:35",
      "platform": "youtube|instagram|x|twitch|tiktok|github|spotify|null",
      "handle": "username_without_at_sign_or_null",
      "relationships": [
        {{"predicate": "worksFor|createdBy|uses|managedBy", "target": "Other Entity Name"}}
      ]
    }}
  ]
}}

Rules:
- Only include entities mentioned substantively (discussed, explained, demonstrated), not passing references
- For people, include full names when available. Skip generic references ("someone", "a guy")
- For tools, include the specific tool/library/framework name, not generic categories
- For accounts, extract from social links in the description and any accounts mentioned in the transcript
- The "platform" and "handle" fields are only for type "account" -- set to null for other types
- If no meaningful entities are found, return {{"entities": []}}
- Timestamps should come from the transcript's timestamp markers
- The video description and channel name provide additional context -- extract entities from those too if substantive
"""


def _slugify(name: str) -> str:
    """Convert a name to a kebab-case filename slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[''']s\b", "s", slug)  # possessives
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:80]


def _detect_platform(url: str) -> str | None:
    """Detect social platform from a URL."""
    for pattern, platform in PLATFORM_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return platform
    return None


def _account_slug(handle: str | None, name: str, platform: str) -> str:
    """Build a slug for an account note: <handle>-<platform> or <name>-<platform>."""
    base = _slugify(handle) if handle else _slugify(name)
    return f"{base}-{platform}"


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
        "Return ONLY valid JSON. No tools, no file access, no conversation. "
        "CRITICAL: Only extract entities that are explicitly named in the "
        "provided transcript or description text. Never invent or infer "
        "entities that do not appear in the input."
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
    description: str | None = None,
    channel: str | None = None,
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
    if channel:
        # Link to account entity if we can determine the platform
        platform = _detect_platform(url) if url else None
        if platform:
            account_slug = _account_slug(None, channel, platform)
            frontmatter["channel"] = f"[[{account_slug}]]"
        else:
            frontmatter["channel"] = channel
    if duration:
        frontmatter["duration"] = round(duration)

    body_lines = [f"# {title}", ""]
    if url:
        body_lines.append(f"Source: {url}")
        body_lines.append("")
    if description:
        body_lines.append("## Description")
        body_lines.append("")
        body_lines.append(description.strip())
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

    # Account entities use a different slug scheme: <handle>-<platform>
    if entity_type == "account":
        return _create_or_update_account(entity, video_slug, video_title)

    slug = _slugify(name)
    folder = ENTITY_FOLDERS[entity_type]
    path = folder / f"{slug}.md"

    timestamp = entity.get("timestamp", "")
    context = entity.get("context", "")
    relationships = entity.get("relationships", [])

    # Build a rich mention block: context + relationships from this video
    mention_parts = [f"**[[{_slugify(video_title)}]]**"]
    if timestamp:
        mention_parts[0] += f" at {timestamp}"
    if context:
        mention_parts.append(f"  {context}")
    for rel in relationships:
        pred = rel.get("predicate", "")
        target = rel.get("target", "")
        if pred and target:
            mention_parts.append(f"  {pred}: [[{_slugify(target)}]]")
    mention_block = "\n".join(mention_parts)

    if path.exists():
        existing = path.read_text()
        video_ref = _slugify(video_title)
        if video_ref in existing:
            return path, "skipped"

        # --- Backfill missing frontmatter relationships ---
        _backfill_relationships(path, entity_type, relationships)

        # Re-read after potential frontmatter changes
        existing = path.read_text()

        # --- Append mention to Video Mentions section ---
        if "## Video Mentions" in existing:
            # Append to existing section
            updated = existing.rstrip() + f"\n\n{mention_block}\n"
        else:
            # Create the section
            updated = existing.rstrip() + f"\n\n## Video Mentions\n\n{mention_block}\n"
        path.write_text(updated)

        # Add mentionedIn to frontmatter if not present
        if "mentionedIn:" not in existing:
            _add_frontmatter_field(path, "mentionedIn", f'"[[{video_ref}]]"')

        return path, "updated"

    # --- Create new note ---
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

    body = f"# {name}\n\n## Video Mentions\n\n{mention_block}\n"
    content = _format_note(frontmatter, body)
    path.write_text(content)
    return path, "created"


# Mapping of entity_type -> {predicate -> frontmatter_key}
_BACKFILL_PREDICATES = {
    "person": {"worksFor": "worksFor"},
    "organization": {},
    "tool": {"createdBy": "createdBy"},
}


def _backfill_relationships(
    path: Path, entity_type: str, relationships: list[dict],
):
    """Add missing relationship fields to an existing note's frontmatter.

    Only adds fields that don't already exist -- never overwrites.
    """
    pred_map = _BACKFILL_PREDICATES.get(entity_type, {})
    if not pred_map:
        return

    existing = path.read_text()
    for rel in relationships:
        predicate = rel.get("predicate", "")
        target = rel.get("target", "")
        fm_key = pred_map.get(predicate)
        if fm_key and target and f"{fm_key}:" not in existing:
            _add_frontmatter_field(
                path, fm_key, f'"[[{_slugify(target)}]]"',
            )


def _create_or_update_account(
    entity: dict,
    video_slug: str,
    video_title: str,
) -> tuple[Path | None, str]:
    """Create or update a vault note for an Account entity.

    Returns (path, action) where action is 'created', 'updated', or 'skipped'.
    """
    name = entity.get("name", "").strip()
    platform = (entity.get("platform") or "").lower().strip()
    handle = (entity.get("handle") or "").strip()
    context = entity.get("context", "")
    relationships = entity.get("relationships", [])

    if not platform:
        return None, "skipped"

    slug = _account_slug(handle, name, platform)
    path = ACCOUNTS_FOLDER / f"{slug}.md"

    mention_line = f"**[[{_slugify(video_title)}]]** -- {context}"

    if path.exists():
        existing = path.read_text()
        video_ref = _slugify(video_title)
        if video_ref in existing:
            return path, "skipped"

        # Backfill managedBy if missing
        for rel in relationships:
            if rel.get("predicate") == "managedBy" and rel.get("target"):
                if "managedBy:" not in existing:
                    _add_frontmatter_field(
                        path, "managedBy",
                        f'"[[{_slugify(rel["target"])}]]"',
                    )

        # Append to Video Mentions section
        existing = path.read_text()  # re-read after potential frontmatter change
        if "## Video Mentions" in existing:
            updated = existing.rstrip() + f"\n\n{mention_line}\n"
        else:
            updated = existing.rstrip() + f"\n\n## Video Mentions\n\n{mention_line}\n"
        path.write_text(updated)
        return path, "updated"

    # Create new account note
    ACCOUNTS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Clean display name (strip platform suffix Sonnet might add)
    display_name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()

    tags = ["account", platform]
    frontmatter = {
        "title": display_name,
        "type": "account",
        "tags": tags,
        "platform": platform,
        "created": str(date.today()),
    }
    if handle:
        frontmatter["handle"] = handle
    # Build managedBy from relationships
    managed_by = []
    for rel in relationships:
        if rel.get("predicate") == "managedBy" and rel.get("target"):
            managed_by.append(f"[[{_slugify(rel['target'])}]]")
    if managed_by:
        if len(managed_by) == 1:
            frontmatter["managedBy"] = managed_by[0]
        else:
            frontmatter["managedBy"] = managed_by

    body = f"# {display_name}\n\n## Video Mentions\n\n{mention_line}\n"
    content = _format_note(frontmatter, body)
    path.write_text(content)
    return path, "created"


def _format_note(frontmatter: dict, body: str) -> str:
    """Format a vault note with YAML frontmatter."""
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            # Check if list contains wiki-links that need quoting
            if value and isinstance(value[0], str) and value[0].startswith("[["):
                quoted = [f'"{v}"' for v in value]
                lines.append(f"{key}: [{', '.join(quoted)}]")
            else:
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
    description: str | None = None,
    channel: str | None = None,
) -> list[dict]:
    """Extract entities from a transcript using Claude Sonnet.

    Args:
        transcript: Full transcript text (may include timestamps).
        title: Video title.
        source_type: youtube, instagram, or local.
        url: Source URL if applicable.
        description: Video description / caption text.
        channel: Channel or uploader name.

    Returns:
        List of entity dicts from Sonnet's extraction.
    """
    prompt = EXTRACTION_PROMPT.format(
        title=title,
        source_type=source_type,
        url=url or "N/A",
        description=description or "N/A",
        channel=channel or "N/A",
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
    description: str | None = None,
    channel: str | None = None,
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
        description: Video description / caption text.
        channel: Channel or uploader name.
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
        description=description, channel=channel,
    )

    if verbose and on_progress:
        on_progress(f"  Found {len(entities)} entities")

    # Create video note
    video_slug = _slugify(title)
    video_path = _create_video_note(
        title, source_type, url, duration, entities,
        description=description, channel=channel,
    )
    if on_progress:
        on_progress(f"  Video note: {video_path.relative_to(VAULT_PATH)}")

    # Ensure the publishing channel has an Account entity
    if channel and source_type in ("youtube", "instagram"):
        platform = source_type
        channel_slug = _account_slug(None, channel, platform)
        channel_account_path = ACCOUNTS_FOLDER / f"{channel_slug}.md"
        if not channel_account_path.exists():
            # Synthesize an account entity for the channel
            channel_entity = {
                "name": channel,
                "type": "account",
                "platform": platform,
                "handle": None,
                "context": f"Publishing channel for this video",
                "relationships": [],
            }
            _create_or_update_account(channel_entity, video_slug, title)
            if on_progress:
                on_progress(f"    + account: {channel} ({channel_slug}.md)")

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
