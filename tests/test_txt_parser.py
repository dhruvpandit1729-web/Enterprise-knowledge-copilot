from __future__ import annotations

from pathlib import Path
import pytest

from app.ingestion.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    UnreadableDocumentError,
)
from app.ingestion.parsers.txt import TxtParser


def test_txt_parser_supported_extensions():
    parser = TxtParser()
    assert parser.supported_extensions == {".txt"}


def test_txt_parser_valid(tmp_path: Path):
    content = "POLICY-ID: POL-401\nEffective: 2026-10-20\nAmount: $10,000.00\nAll staff must comply."
    txt_path = tmp_path / "policy.txt"
    txt_path.write_text(content, encoding="utf-8")

    parser = TxtParser()
    doc = parser.parse(txt_path)

    assert doc.source == "policy.txt"
    assert doc.metadata["doc_type"] == "txt"
    assert len(doc.sections) == 1

    section = doc.sections[0]
    assert section.page_number is None  # Do not invent page numbers
    assert section.heading is None
    assert "POLICY-ID: POL-401" in section.text
    assert "Effective: 2026-10-20" in section.text
    assert "Amount: $10,000.00" in section.text


def test_txt_parser_utf8_with_bom():
    raw_bytes = b"\xef\xbb\xbfUTF-8 text with Byte Order Mark."
    parser = TxtParser()
    doc = parser.parse_bytes(raw_bytes, filename="bom.txt")

    assert len(doc.sections) == 1
    assert doc.sections[0].text == "UTF-8 text with Byte Order Mark."
    assert not doc.sections[0].text.startswith("\ufeff")


def test_txt_parser_empty_bytes(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_bytes(b"")

    parser = TxtParser()
    with pytest.raises(EmptyDocumentError, match="empty \\(0 bytes\\)"):
        parser.parse(empty_file)

    with pytest.raises(EmptyDocumentError, match="empty \\(0 bytes\\)"):
        parser.parse_bytes(b"", filename="empty.txt")


def test_txt_parser_whitespace_only(tmp_path: Path):
    ws_file = tmp_path / "whitespace.txt"
    ws_file.write_text("   \n\t  \r\n   ", encoding="utf-8")

    parser = TxtParser()
    with pytest.raises(EmptyDocumentError, match="empty or contains only whitespace"):
        parser.parse(ws_file)


def test_txt_parser_invalid_utf8():
    invalid_bytes = b"\x80\x81\xff\xfe\xab"
    parser = TxtParser()
    with pytest.raises(UnreadableDocumentError, match="Failed to decode TXT file"):
        parser.parse_bytes(invalid_bytes, filename="invalid.txt")


def test_txt_parser_file_not_found():
    parser = TxtParser()
    with pytest.raises(DocumentNotFoundError):
        parser.parse(Path("missing_file.txt"))

