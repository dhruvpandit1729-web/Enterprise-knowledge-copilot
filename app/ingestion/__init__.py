from __future__ import annotations

from app.ingestion.base import BaseParser
from app.ingestion.chunker import DocumentChunker, chunk_document, compute_content_hash
from app.ingestion.dispatcher import DocumentDispatcher, get_parser, parse_document
from app.ingestion.embedder import (
    EmbeddingService,
    get_default_embedder,
    set_default_embedder,
)
from app.ingestion.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    IngestionError,
    UnreadableDocumentError,
    UnsupportedFormatError,
)
from app.ingestion.normalizer import normalize_text
from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.markdown import MarkdownParser
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.parsers.txt import TxtParser
from app.schemas.document import DocumentChunk, ExtractedDocument, ExtractedSection

__all__ = [
    "BaseParser",
    "DocxParser",
    "DocumentChunk",
    "DocumentChunker",
    "DocumentDispatcher",
    "DocumentNotFoundError",
    "EmbeddingService",
    "EmptyDocumentError",
    "ExtractedDocument",
    "ExtractedSection",
    "IngestionError",
    "MarkdownParser",
    "PDFParser",
    "TxtParser",
    "UnreadableDocumentError",
    "UnsupportedFormatError",
    "chunk_document",
    "compute_content_hash",
    "get_default_embedder",
    "get_parser",
    "normalize_text",
    "parse_document",
    "set_default_embedder",
]
