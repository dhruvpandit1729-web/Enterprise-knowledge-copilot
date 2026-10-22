from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ExtractedSection(BaseModel):
    """Represents a discrete extracted unit of text and its source metadata."""

    text: str
    page_number: int | None = None
    heading: str | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedDocument(BaseModel):
    """Represents an extracted document with its sections and source metadata."""

    source: str
    sections: list[ExtractedSection]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Returns the combined text of all sections separated by double newlines."""
        return "\n\n".join(section.text for section in self.sections if section.text.strip())

    @property
    def page_count(self) -> int | None:
        """Returns the highest page number if page numbers are present, else None."""
        page_nums = [s.page_number for s in self.sections if s.page_number is not None]
        return max(page_nums) if page_nums else None


class DocumentChunk(BaseModel):
    """Represents a tokenizer-bounded chunk of a document for retrieval and embedding."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    token_count: int
    section_index: int
    chunk_index: int
    page_number: int | None = None
    heading: str | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
