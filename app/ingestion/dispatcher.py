from __future__ import annotations

from pathlib import Path

from app.ingestion.base import BaseParser
from app.ingestion.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    UnreadableDocumentError,
    UnsupportedFormatError,
)
from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.markdown import MarkdownParser
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.parsers.txt import TxtParser
from app.schemas.document import ExtractedDocument


class DocumentDispatcher:
    """Dispatch document extraction requests to the appropriate parser."""

    def __init__(self, parsers: list[BaseParser] | None = None) -> None:
        self._parsers: dict[str, BaseParser] = {}

        registered = parsers or [
            PDFParser(),
            DocxParser(),
            TxtParser(),
            MarkdownParser(),
        ]

        for parser in registered:
            self.register_parser(parser)

    def register_parser(self, parser: BaseParser) -> None:
        """Register a parser for its supported file extensions."""
        for ext in parser.supported_extensions:
            self._parsers[ext.lower()] = parser

    @property
    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""
        return set(self._parsers.keys())

    def get_parser(self, filename_or_path: str | Path) -> BaseParser:
        """Get the parser registered for the given file extension."""
        path = Path(filename_or_path)
        suffix = path.suffix.lower().strip()

        if not suffix or suffix not in self._parsers:
            supported = ", ".join(sorted(self._parsers.keys()))
            display_ext = (
                suffix
                if suffix
                else f"(no extension in '{path.name}')"
            )
            raise UnsupportedFormatError(
                f"Unsupported file format '{display_ext}'. "
                f"Supported formats are: {supported}"
            )

        return self._parsers[suffix]

    def parse_file(
        self,
        file_path: str | Path,
        original_filename: str | None = None,
    ) -> ExtractedDocument:
        """
        Parse a document from a file path on disk.

        If original_filename is provided, preserve it as the document's
        source metadata instead of using the temporary filesystem name.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise DocumentNotFoundError(
                f"File not found: '{path}'"
            )

        if not path.is_file():
            raise UnreadableDocumentError(
                f"Path is not a regular file: '{path}'"
            )

        if path.stat().st_size == 0:
            raise EmptyDocumentError(
                f"File '{path.name}' is empty (0 bytes)."
            )

        # Select the parser using the file extension.
        parser = self.get_parser(path)

        # Parsers accept a path only. Do not pass filename= here.
        document = parser.parse(path)

        # Restore the original uploaded filename in the extracted metadata.
        if original_filename:
            safe_filename = Path(
                original_filename.replace("\\", "/")
            ).name

            if safe_filename:
                document.source = safe_filename

        return document

    def parse_bytes(
        self,
        content: bytes,
        filename: str,
    ) -> ExtractedDocument:
        """Parse a document directly from in-memory bytes."""
        if not content:
            raise EmptyDocumentError(
                f"File '{filename}' is empty (0 bytes)."
            )

        parser = self.get_parser(filename)

        return parser.parse_bytes(
            content,
            filename=filename,
        )

    def parse_document(
        self,
        source: str | Path,
        content: bytes | None = None,
    ) -> ExtractedDocument:
        """Parse from bytes when supplied; otherwise parse from a file."""
        if content is not None:
            filename = Path(source).name if source else "document"
            return self.parse_bytes(
                content,
                filename=filename,
            )

        return self.parse_file(source)


_default_dispatcher = DocumentDispatcher()


def get_parser(filename_or_path: str | Path) -> BaseParser:
    """Convenience function to get the registered parser."""
    return _default_dispatcher.get_parser(filename_or_path)


def parse_document(
    source: str | Path,
    content: bytes | None = None,
) -> ExtractedDocument:
    """Convenience function for parsing a file or byte content."""
    return _default_dispatcher.parse_document(
        source,
        content=content,
    )