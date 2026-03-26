"""Tests for entity extraction helpers.

Tests the pure functions in extract.py: slugification, YAML formatting,
JSON response parsing, timestamp resolution, and frontmatter manipulation.
"""

import os
import tempfile
from pathlib import Path

from glean.extract import (
    _slugify,
    _format_note,
    _parse_extraction_response,
    _find_chunk_heading,
    _add_frontmatter_field,
    _detect_platform,
    _account_slug,
    _yaml_escape,
    _fmt_heading_time,
)


class TestSlugify:
    def test_basic(self):
        assert _slugify("Rick Astley") == "rick-astley"

    def test_special_characters(self):
        assert _slugify("Tool: A Framework!") == "tool-a-framework"

    def test_possessives(self):
        assert _slugify("Astley's Greatest") == "astleys-greatest"

    def test_multiple_spaces(self):
        assert _slugify("too   many    spaces") == "too-many-spaces"

    def test_leading_trailing(self):
        assert _slugify("  -hello world-  ") == "hello-world"

    def test_truncation(self):
        long_name = "a" * 100
        assert len(_slugify(long_name)) <= 80

    def test_unicode_stripped(self):
        # Combining accent is stripped but base letter e survives
        assert _slugify("cafe\u0301 du monde") == "cafe-du-monde"


class TestYamlEscape:
    def test_plain_string(self):
        assert _yaml_escape("hello world") == "hello world"

    def test_colon_in_value(self):
        assert _yaml_escape("My Talk: A Deep Dive") == '"My Talk: A Deep Dive"'

    def test_hash_in_value(self):
        assert _yaml_escape("C# Programming") == '"C# Programming"'

    def test_quotes_escaped(self):
        result = _yaml_escape('She said "hi"')
        assert result == '"She said \\"hi\\""'

    def test_leading_dash(self):
        assert _yaml_escape("- a list item").startswith('"')

    def test_wiki_link_not_affected(self):
        # Wiki-links are handled separately in _format_note, but _yaml_escape
        # should still quote them since they contain brackets
        result = _yaml_escape("[[some-link]]")
        assert result.startswith('"')


class TestFormatNote:
    def test_basic_frontmatter(self):
        fm = {"title": "Test", "type": "person", "tags": ["contact", "person"]}
        result = _format_note(fm, "# Test\n")
        assert result.startswith("---\n")
        assert "title: Test" in result
        assert "tags: [contact, person]" in result
        assert result.endswith("# Test\n")

    def test_wiki_link_quoted(self):
        fm = {"worksFor": "[[some-org]]"}
        result = _format_note(fm, "body")
        assert 'worksFor: "[[some-org]]"' in result

    def test_wiki_link_array(self):
        fm = {"managedBy": ["[[person-a]]", "[[person-b]]"]}
        result = _format_note(fm, "body")
        assert 'managedBy: ["[[person-a]]", "[[person-b]]"]' in result

    def test_yaml_special_chars_quoted(self):
        fm = {"title": "My Talk: A Deep Dive", "type": "video"}
        result = _format_note(fm, "body")
        assert 'title: "My Talk: A Deep Dive"' in result

    def test_numeric_value(self):
        fm = {"duration": 212}
        result = _format_note(fm, "body")
        assert "duration: 212" in result


class TestParseExtractionResponse:
    def test_clean_json(self):
        response = '{"entities": [{"name": "Rick", "type": "person"}]}'
        entities = _parse_extraction_response(response)
        assert len(entities) == 1
        assert entities[0]["name"] == "Rick"

    def test_code_fenced(self):
        response = '```json\n{"entities": [{"name": "Rick", "type": "person"}]}\n```'
        entities = _parse_extraction_response(response)
        assert len(entities) == 1

    def test_empty_entities(self):
        response = '{"entities": []}'
        assert _parse_extraction_response(response) == []

    def test_none_input(self):
        assert _parse_extraction_response(None) == []

    def test_garbage_input(self):
        assert _parse_extraction_response("not json at all") == []

    def test_json_with_surrounding_text(self):
        response = 'Here is the result:\n{"entities": [{"name": "Rick", "type": "person"}]}\nDone.'
        entities = _parse_extraction_response(response)
        assert len(entities) == 1
        assert entities[0]["name"] == "Rick"

    def test_nested_braces_handled(self):
        response = '{"entities": [{"name": "Rick", "type": "person", "relationships": [{"predicate": "worksFor", "target": "PWL"}]}]}'
        entities = _parse_extraction_response(response)
        assert len(entities) == 1
        assert entities[0]["relationships"][0]["target"] == "PWL"


class TestFindChunkHeading:
    CHUNKS = [
        {"start_time": 0, "end_time": 30},
        {"start_time": 30, "end_time": 60},
        {"start_time": 60, "end_time": 90},
    ]

    def test_finds_correct_chunk(self):
        heading = _find_chunk_heading("00:00:15", self.CHUNKS)
        assert heading == "00:00:00 - 00:00:30"

    def test_second_chunk(self):
        heading = _find_chunk_heading("00:00:45", self.CHUNKS)
        assert heading == "00:00:30 - 00:01:00"

    def test_zero_timestamp_fallback(self):
        heading = _find_chunk_heading("00:00:00", self.CHUNKS)
        assert heading == "00:00:00 - 00:00:30"

    def test_out_of_range(self):
        heading = _find_chunk_heading("00:05:00", self.CHUNKS)
        assert heading is None

    def test_empty_chunks(self):
        assert _find_chunk_heading("00:00:15", []) is None

    def test_none_timestamp(self):
        assert _find_chunk_heading(None, self.CHUNKS) is None

    def test_bad_timestamp_format(self):
        assert _find_chunk_heading("invalid", self.CHUNKS) is None


class TestAddFrontmatterField:
    def test_adds_field(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("---\ntitle: Test\ntype: person\n---\n\n# Test\n")
            f.flush()
            path = Path(f.name)
        try:
            _add_frontmatter_field(path, "worksFor", '"[[some-org]]"')
            content = path.read_text()
            assert 'worksFor: "[[some-org]]"' in content
            assert content.count("---") == 2  # opening and closing
        finally:
            os.unlink(path)

    def test_does_not_overwrite_existing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("---\ntitle: Test\nworksFor: existing\n---\n\n# Test\n")
            f.flush()
            path = Path(f.name)
        try:
            _add_frontmatter_field(path, "worksFor", '"[[new-org]]"')
            content = path.read_text()
            assert "existing" in content
            assert "new-org" not in content
        finally:
            os.unlink(path)

    def test_body_horizontal_rule_not_split(self):
        """Body containing --- should not confuse frontmatter parsing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("---\ntitle: Test\n---\n\n# Test\n\n---\n\nMore content\n")
            f.flush()
            path = Path(f.name)
        try:
            _add_frontmatter_field(path, "newField", "value")
            content = path.read_text()
            assert "newField: value" in content
            # Body should still contain the horizontal rule
            assert "More content" in content
        finally:
            os.unlink(path)


class TestDetectPlatform:
    def test_youtube(self):
        assert _detect_platform("https://www.youtube.com/watch?v=abc") == "youtube"

    def test_youtu_be(self):
        assert _detect_platform("https://youtu.be/abc") == "youtube"

    def test_instagram(self):
        assert _detect_platform("https://www.instagram.com/reels/abc/") == "instagram"

    def test_unknown(self):
        assert _detect_platform("https://example.com") is None


class TestAccountSlug:
    def test_with_handle(self):
        assert _account_slug("ludwigahgren", "Ludwig", "youtube") == "ludwigahgren-youtube"

    def test_without_handle(self):
        assert _account_slug(None, "Ludwig", "youtube") == "ludwig-youtube"


class TestFmtHeadingTime:
    def test_zero(self):
        assert _fmt_heading_time(0) == "00:00:00"

    def test_minutes(self):
        assert _fmt_heading_time(90) == "00:01:30"

    def test_hours(self):
        assert _fmt_heading_time(3661) == "01:01:01"
