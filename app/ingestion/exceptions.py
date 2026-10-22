from __future__ import annotations


class IngestionError(Exception):
    """Base exception for all document ingestion and extraction errors."""


class UnsupportedFormatError(IngestionError):
    """Raised when an unsupported file format or extension is encountered."""


class EmptyDocumentError(IngestionError):
    """Raised when a document is 0 bytes or contains no extractable text."""


class UnreadableDocumentError(IngestionError):
    """Raised when a document cannot be parsed or read due to corruption or format errors."""


class DocumentNotFoundError(IngestionError, FileNotFoundError):
    """Raised when the specified file path does not exist on disk."""
