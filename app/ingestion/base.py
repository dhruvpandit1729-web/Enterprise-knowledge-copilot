from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.document import ExtractedDocument


class BaseParser(ABC):
    """Abstract base class for document format parsers."""

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Set of supported lowercase file extensions including leading dot (e.g. {'.pdf'})."""
        pass

    @abstractmethod
    def parse(self, file_path: str | Path) -> ExtractedDocument:
        """Parse a document from a file on disk."""
        pass

    @abstractmethod
    def parse_bytes(self, content: bytes, filename: str) -> ExtractedDocument:
        """Parse a document from in-memory bytes."""
        pass
