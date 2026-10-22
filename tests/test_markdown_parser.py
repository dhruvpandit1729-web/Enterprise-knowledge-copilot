from __future__ import annotations

from pathlib import Path
import pytest

from app.ingestion.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    UnreadableDocumentError,
)
from app.ingestion.parsers.markdown import MarkdownParser


def test_markdown_parser_supported_extensions():
    parser = MarkdownParser()
    assert parser.supported_extensions == {".md", ".markdown"}


def test_markdown_parser_valid_with_headings(tmp_path: Path):
    md_content = """# Company Handbook

Welcome to the company.

## Information Security

All employees must lock workstations.

```python
# This is a code comment, NOT a heading
token = "abc123"
```

## Work From Home

Eligible after 90 days.
"""
    md_path = tmp_path / "handbook.md"
    md_path.write_text(md_content, encoding="utf-8")

    parser = MarkdownParser()
    doc = parser.parse(md_path)

    assert doc.source == "handbook.md"
    assert doc.metadata["doc_type"] == "markdown"
    assert len(doc.sections) == 3

    # Section 1: Company Handbook
    s1 = doc.sections[0]
    assert s1.page_number is None  # Do not invent page numbers
    assert s1.heading == "Company Handbook"
    assert "Welcome to the company." in s1.text

    # Section 2: Information Security
    s2 = doc.sections[1]
    assert s2.page_number is None
    assert s2.heading == "Company Handbook > Information Security"
    assert "All employees must lock workstations." in s2.text
    # Confirm code block with # was preserved in text and NOT treated as a heading
    assert "# This is a code comment, NOT a heading" in s2.text

    # Section 3: Work From Home
    s3 = doc.sections[2]
    assert s3.page_number is None
    assert s3.heading == "Company Handbook > Work From Home"
    assert "Eligible after 90 days." in s3.text


def test_markdown_parser_setext_headings():
    md_content = """Document Title
==============

Intro paragraph.

Section One
-----------

Content for section one.
"""
    parser = MarkdownParser()
    doc = parser.parse_bytes(md_content.encode("utf-8"), filename="setext.markdown")

    assert len(doc.sections) == 2
    assert doc.sections[0].heading == "Document Title"
    assert "Intro paragraph." in doc.sections[0].text
    assert doc.sections[1].heading == "Document Title > Section One"
    assert "Content for section one." in doc.sections[1].text


def test_markdown_parser_no_headings():
    content = "Just plain text\nwithout any markdown headings."
    parser = MarkdownParser()
    doc = parser.parse_bytes(content.encode("utf-8"), filename="no_headings.md")

    assert len(doc.sections) == 1
    assert doc.sections[0].heading is None
    assert doc.sections[0].page_number is None
    assert doc.sections[0].text == "Just plain text\nwithout any markdown headings."


def test_markdown_parser_empty_bytes(tmp_path: Path):
    empty_file = tmp_path / "empty.md"
    empty_file.write_bytes(b"")

    parser = MarkdownParser()
    with pytest.raises(EmptyDocumentError, match="empty \\(0 bytes\\)"):
        parser.parse(empty_file)

    with pytest.raises(EmptyDocumentError, match="empty \\(0 bytes\\)"):
        parser.parse_bytes(b"", filename="empty.md")


def test_markdown_parser_whitespace_only(tmp_path: Path):
    ws_file = tmp_path / "whitespace.md"
    ws_file.write_text("   \n\t  \n  ", encoding="utf-8")

    parser = MarkdownParser()
    with pytest.raises(EmptyDocumentError, match="empty or contains only whitespace"):
        parser.parse(ws_file)


def test_markdown_parser_invalid_utf8():
    invalid_bytes = b"\xfe\xff\x80\x81invalid"
    parser = MarkdownParser()
    with pytest.raises(UnreadableDocumentError, match="Failed to decode Markdown file"):
        parser.parse_bytes(invalid_bytes, filename="invalid.md")


def test_markdown_parser_file_not_found():
    parser = MarkdownParser()
    with pytest.raises(DocumentNotFoundError):
        parser.parse(Path("missing_file.md"))
