from __future__ import annotations

import io
from pathlib import Path
import pytest
import pypdf

from app.ingestion.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    UnreadableDocumentError,
)
from app.ingestion.parsers.pdf import PDFParser


def build_pdf_bytes(pages: list[str]) -> bytes:
    """Helper to build a valid multi-page PDF in-memory with the given text per page."""
    num_pages = len(pages)
    font_obj_num = 3 + 2 * num_pages
    kids = [f"{3 + 2 * i} 0 R" for i in range(num_pages)]

    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        f"2 0 obj\n<< /Type /Pages /Kids [{chr(32).join(kids)}] /Count {num_pages} >>\nendobj\n",
    ]

    for i, text in enumerate(pages):
        page_obj_num = 3 + 2 * i
        content_obj_num = page_obj_num + 1
        stream_content = f"BT\n/F1 12 Tf\n72 712 Td\n({text}) Tj\nET"
        content_bytes = stream_content.encode("latin1")

        objects.append(
            f"{page_obj_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
            f"/Contents {content_obj_num} 0 R >>\nendobj\n"
        )
        objects.append(
            f"{content_obj_num} 0 obj\n"
            f"<< /Length {len(content_bytes)} >>\n"
            f"stream\n{stream_content}\nendstream\nendobj\n"
        )

    objects.append(
        f"{font_obj_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    pdf_data = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf_data.encode("latin1")))
        pdf_data += obj

    xref_offset = len(pdf_data.encode("latin1"))
    xref = f"xref\n0 {len(offsets)}\n0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n"

    trailer = f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    pdf_data += xref + trailer
    return pdf_data.encode("latin1")


def test_pdf_parser_supported_extensions():
    parser = PDFParser()
    assert parser.supported_extensions == {".pdf"}


def test_pdf_parser_valid_multi_page(tmp_path: Path):
    pdf_bytes = build_pdf_bytes(["First page policy text", "Second page rules"])
    pdf_path = tmp_path / "policy.pdf"
    pdf_path.write_bytes(pdf_bytes)

    parser = PDFParser()
    doc = parser.parse(pdf_path)

    assert doc.source == "policy.pdf"
    assert doc.metadata["doc_type"] == "pdf"
    assert doc.metadata["total_pages"] == 2
    assert doc.page_count == 2
    assert len(doc.sections) == 2

    # Check accurate 1-based page numbers
    sec1 = doc.sections[0]
    assert sec1.page_number == 1
    assert sec1.heading is None
    assert sec1.text == "First page policy text"

    sec2 = doc.sections[1]
    assert sec2.page_number == 2
    assert sec2.heading is None
    assert sec2.text == "Second page rules"

    assert "First page policy text" in doc.full_text
    assert "Second page rules" in doc.full_text


def test_pdf_parser_bytes():
    pdf_bytes = build_pdf_bytes(["Memory PDF text"])
    parser = PDFParser()
    doc = parser.parse_bytes(pdf_bytes, filename="in_memory.pdf")

    assert doc.source == "in_memory.pdf"
    assert len(doc.sections) == 1
    assert doc.sections[0].page_number == 1
    assert doc.sections[0].text == "Memory PDF text"


def test_pdf_parser_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    parser = PDFParser()
    with pytest.raises(EmptyDocumentError, match="empty \\(0 bytes\\)"):
        parser.parse(empty_file)

    with pytest.raises(EmptyDocumentError, match="empty \\(0 bytes\\)"):
        parser.parse_bytes(b"", filename="empty.pdf")


def test_pdf_parser_blank_pages():
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    blank_bytes = buf.getvalue()

    parser = PDFParser()
    with pytest.raises(EmptyDocumentError, match="contains no extractable text"):
        parser.parse_bytes(blank_bytes, filename="blank.pdf")


def test_pdf_parser_corrupt_file(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"Not a real PDF stream")

    parser = PDFParser()
    with pytest.raises(UnreadableDocumentError):
        parser.parse(corrupt_file)

    with pytest.raises(UnreadableDocumentError):
        parser.parse_bytes(b"Corrupt data", filename="corrupt.pdf")


def test_pdf_parser_encrypted():
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("topsecret")
    buf = io.BytesIO()
    writer.write(buf)

    parser = PDFParser()
    with pytest.raises(UnreadableDocumentError, match="password-protected or encrypted"):
        parser.parse_bytes(buf.getvalue(), filename="locked.pdf")


def test_pdf_parser_file_not_found():
    parser = PDFParser()
    with pytest.raises(DocumentNotFoundError):
        parser.parse(Path("non_existent_file.pdf"))
