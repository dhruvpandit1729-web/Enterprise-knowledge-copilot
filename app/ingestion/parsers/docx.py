from __future__ import annotations

import io
from pathlib import Path
import re
import zipfile

import docx
from docx.opc.exceptions import OpcError, PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.base import BaseParser
from app.ingestion.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    UnreadableDocumentError,
)
from app.ingestion.normalizer import normalize_text
from app.schemas.document import ExtractedDocument, ExtractedSection


class DocxParser(BaseParser):
    """Parser for DOCX documents using python-docx, retaining heading information."""

    @property
    def supported_extensions(self) -> set[str]:
        return {".docx"}

    def parse(self, file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path).resolve()
        if not path.exists():
            raise DocumentNotFoundError(f"File not found: '{path}'")
        if not path.is_file():
            raise UnreadableDocumentError(f"Path is not a regular file: '{path}'")
        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"DOCX file '{path.name}' is empty (0 bytes).")

        try:
            with open(path, "rb") as f:
                content = f.read()
        except OSError as exc:
            raise UnreadableDocumentError(f"Unable to read DOCX file '{path.name}': {exc}") from exc

        return self.parse_bytes(content, filename=path.name)

    def parse_bytes(self, content: bytes, filename: str) -> ExtractedDocument:
        if not content:
            raise EmptyDocumentError(f"DOCX file '{filename}' is empty (0 bytes).")

        try:
            doc = docx.Document(io.BytesIO(content))
        except (PackageNotFoundError, OpcError, zipfile.BadZipFile) as exc:
            raise UnreadableDocumentError(f"Corrupt or invalid DOCX file '{filename}': {exc}") from exc
        except Exception as exc:
            raise UnreadableDocumentError(f"Failed to parse DOCX '{filename}': {exc}") from exc

        sections: list[ExtractedSection] = []
        heading_stack: dict[int, str] = {}
        current_heading_path: str | None = None
        current_block_lines: list[str] = []

        def _flush_current_block() -> None:
            nonlocal current_block_lines
            if not current_block_lines:
                return
            combined_raw = "\n\n".join(current_block_lines)
            normalized = normalize_text(combined_raw)
            if normalized:
                sections.append(
                    ExtractedSection(
                        text=normalized,
                        page_number=None,  # Do not invent page numbers
                        heading=current_heading_path,
                        metadata={
                            "heading": current_heading_path,
                        }
                        if current_heading_path
                        else {},
                    )
                )
            current_block_lines = []

        for child in doc.element.body:
            if child.tag.endswith("p"):
                p = Paragraph(child, doc)
                p_text = normalize_text(p.text)
                if not p_text:
                    continue

                style_name = (p.style.name or "").strip() if p.style else ""
                is_heading, level = self._detect_heading_level(style_name)

                if is_heading:
                    # Flush previous block before switching to new heading
                    _flush_current_block()

                    # Update heading hierarchy stack
                    heading_stack = {lvl: text for lvl, text in heading_stack.items() if lvl < level}
                    heading_stack[level] = p_text
                    current_heading_path = " > ".join(
                        heading_stack[lvl] for lvl in sorted(heading_stack.keys())
                    )
                    current_block_lines.append(p_text)
                else:
                    current_block_lines.append(p_text)

            elif child.tag.endswith("tbl"):
                t = Table(child, doc)
                table_lines: list[str] = []
                for row in t.rows:
                    row_cells = [normalize_text(cell.text) for cell in row.cells]
                    row_text = " | ".join(c for c in row_cells if c)
                    if row_text:
                        table_lines.append(row_text)
                if table_lines:
                    current_block_lines.append("\n".join(table_lines))

        # Flush any remaining block
        _flush_current_block()

        if not sections:
            raise EmptyDocumentError(
                f"DOCX document '{filename}' contains no extractable text."
            )

        return ExtractedDocument(
            source=filename,
            sections=sections,
            metadata={
                "doc_type": "docx",
                "section_count": len(sections),
            },
        )

    @staticmethod
    def _detect_heading_level(style_name: str) -> tuple[bool, int]:
        """Detect whether a paragraph style represents a heading and return (is_heading, level)."""
        lower = style_name.lower().strip()
        if not lower:
            return False, 0

        if lower == "title":
            return True, 0
        if lower == "subtitle":
            return True, 1

        match = re.match(r"^heading\s*(\d+)$", lower)
        if match:
            return True, int(match.group(1))

        if lower.startswith("heading"):
            return True, 1

        return False, 0
