from __future__ import annotations

import io
from pathlib import Path
import pypdf
import pypdf.errors

from app.ingestion.base import BaseParser
from app.ingestion.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    UnreadableDocumentError,
)
from app.ingestion.normalizer import normalize_text
from app.schemas.document import ExtractedDocument, ExtractedSection


class PDFParser(BaseParser):
    """Parser for PDF documents using pypdf, preserving accurate 1-based page numbers."""

    @property
    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def parse(self, file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path).resolve()
        if not path.exists():
            raise DocumentNotFoundError(f"File not found: '{path}'")
        if not path.is_file():
            raise UnreadableDocumentError(f"Path is not a regular file: '{path}'")
        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"PDF file '{path.name}' is empty (0 bytes).")

        try:
            with open(path, "rb") as f:
                content = f.read()
        except OSError as exc:
            raise UnreadableDocumentError(f"Unable to read PDF file '{path.name}': {exc}") from exc

        return self.parse_bytes(content, filename=path.name)

    def parse_bytes(self, content: bytes, filename: str) -> ExtractedDocument:
        if not content:
            raise EmptyDocumentError(f"PDF file '{filename}' is empty (0 bytes).")

        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
        except (pypdf.errors.PdfReadError, pypdf.errors.EmptyFileError) as exc:
            raise UnreadableDocumentError(f"Failed to parse PDF '{filename}': {exc}") from exc
        except Exception as exc:
            raise UnreadableDocumentError(f"Corrupt or invalid PDF '{filename}': {exc}") from exc

        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
                if decrypted == pypdf.PasswordType.NOT_DECRYPTED:
                    raise UnreadableDocumentError(
                        f"PDF '{filename}' is password-protected or encrypted."
                    )
            except Exception as exc:
                raise UnreadableDocumentError(
                    f"PDF '{filename}' is encrypted and cannot be read: {exc}"
                ) from exc

        if len(reader.pages) == 0:
            raise EmptyDocumentError(f"PDF '{filename}' contains no pages.")

        sections: list[ExtractedSection] = []
        for idx, page in enumerate(reader.pages):
            page_number = idx + 1  # 1-based page number
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                raise UnreadableDocumentError(
                    f"Failed to extract text from page {page_number} of PDF '{filename}': {exc}"
                ) from exc

            normalized = normalize_text(raw_text)
            if normalized:
                sections.append(
                    ExtractedSection(
                        text=normalized,
                        page_number=page_number,
                        heading=None,
                        metadata={"page_number": page_number},
                    )
                )

        if not sections:
            raise EmptyDocumentError(
                f"PDF '{filename}' contains no extractable text across {len(reader.pages)} page(s)."
            )

        return ExtractedDocument(
            source=filename,
            sections=sections,
            metadata={
                "doc_type": "pdf",
                "total_pages": len(reader.pages),
                "extracted_pages": len(sections),
            },
        )
