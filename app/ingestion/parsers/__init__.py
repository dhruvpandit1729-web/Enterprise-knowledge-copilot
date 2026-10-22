from __future__ import annotations

from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.markdown import MarkdownParser
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.parsers.txt import TxtParser

__all__ = ["DocxParser", "MarkdownParser", "PDFParser", "TxtParser"]
