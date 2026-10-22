from __future__ import annotations

from pathlib import Path

from app.ingestion.base import BaseParser
from app.ingestion.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    UnreadableDocumentError,
)
from app.ingestion.normalizer import normalize_text
from app.schemas.document import ExtractedDocument, ExtractedSection


class TxtParser(BaseParser):
    """Parser for plain text documents using UTF-8 extraction."""

    @property
    def supported_extensions(self) -> set[str]:
        return {".txt"}

    def parse(self, file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path).resolve()
        if not path.exists():
            raise DocumentNotFoundError(f"File not found: '{path}'")
        if not path.is_file():
            raise UnreadableDocumentError(f"Path is not a regular file: '{path}'")
        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"TXT file '{path.name}' is empty (0 bytes).")

        try:
            with open(path, "rb") as f:
                content = f.read()
        except OSError as exc:
            raise UnreadableDocumentError(f"Unable to read TXT file '{path.name}': {exc}") from exc

        return self.parse_bytes(content, filename=path.name)

    def parse_bytes(self, content: bytes, filename: str) -> ExtractedDocument:
        if not content:
            raise EmptyDocumentError(f"TXT file '{filename}' is empty (0 bytes).")

        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UnreadableDocumentError(
                f"Failed to decode TXT file '{filename}' with UTF-8: {exc}"
            ) from exc

        normalized = normalize_text(text)
        if not normalized:
            raise EmptyDocumentError(
                f"TXT document '{filename}' is empty or contains only whitespace."
            )

        section = ExtractedSection(
            text=normalized,
            page_number=None,  # Do not invent page numbers
            heading=None,
        )

        return ExtractedDocument(
            source=filename,
            sections=[section],
            metadata={
                "doc_type": "txt",
                "byte_size": len(content),
            },
        )
