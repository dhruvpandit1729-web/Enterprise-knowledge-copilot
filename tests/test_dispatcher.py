from __future__ import annotations

from pathlib import Path
import pytest

from app.ingestion.dispatcher import DocumentDispatcher, get_parser, parse_document
from app.ingestion.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    UnsupportedFormatError,
)
from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.markdown import MarkdownParser
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.parsers.txt import TxtParser


def test_dispatcher_supported_extensions():
    dispatcher = DocumentDispatcher()
    assert dispatcher.supported_extensions == {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".markdown",
    }


@pytest.mark.parametrize(
    "filename, expected_cls",
    [
        ("report.pdf", PDFParser),
        ("report.PDF", PDFParser),
        ("handbook.docx", DocxParser),
        ("handbook.DOCX", DocxParser),
        ("notes.txt", TxtParser),
        ("notes.TXT", TxtParser),
        ("guide.md", MarkdownParser),
        ("guide.MD", MarkdownParser),
        ("readme.markdown", MarkdownParser),
        ("readme.MARKDOWN", MarkdownParser),
    ],
)
def test_dispatcher_resolution(filename, expected_cls):
    parser = get_parser(filename)
    assert isinstance(parser, expected_cls)


@pytest.mark.parametrize(
    "unsupported_name",
    [
        ("archive.zip"),
        ("document.doc"),  # Legacy binary doc
        ("webpage.html"),
        ("script.py"),
        ("data.json"),
        ("table.csv"),
        ("executable.exe"),
        ("no_extension_file"),
        (".gitkeep"),
    ],
)
def test_dispatcher_rejects_unsupported_formats(unsupported_name):
    with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
        get_parser(unsupported_name)


def test_parse_document_from_disk(tmp_path: Path):
    file_path = tmp_path / "valid.txt"
    file_path.write_text("Dispatched file content", encoding="utf-8")

    doc = parse_document(file_path)
    assert doc.source == "valid.txt"
    assert doc.sections[0].text == "Dispatched file content"


def test_parse_document_from_bytes():
    content = b"# Dispatched from memory\n\nContent here."
    doc = parse_document("memory.md", content=content)
    assert doc.source == "memory.md"
    assert doc.sections[0].heading == "Dispatched from memory"


def test_parse_document_missing_file():
    with pytest.raises(DocumentNotFoundError):
        parse_document("non_existent_path.txt")


def test_parse_document_empty_file(tmp_path: Path):
    empty_path = tmp_path / "zero_bytes.txt"
    empty_path.write_bytes(b"")

    with pytest.raises(EmptyDocumentError):
        parse_document(empty_path)

    with pytest.raises(EmptyDocumentError):
        parse_document("zero_bytes.txt", content=b"")
