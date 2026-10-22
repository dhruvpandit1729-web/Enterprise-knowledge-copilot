from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from app.core.config import settings
from app.ingestion.exceptions import EmptyDocumentError
from app.schemas.document import DocumentChunk, ExtractedDocument, ExtractedSection

_global_tokenizer: Any | None = None


def get_default_tokenizer(model_name: str | None = None) -> Any:
    """Lazily loads and reuses the AutoTokenizer for the configured embedding model."""
    global _global_tokenizer
    if _global_tokenizer is None:
        target_model = model_name or settings.embed_model
        from transformers import AutoTokenizer

        _global_tokenizer = AutoTokenizer.from_pretrained(target_model)
    return _global_tokenizer


def set_default_tokenizer(tokenizer: Any | None) -> None:
    """Sets or resets the cached default tokenizer (useful for test isolation)."""
    global _global_tokenizer
    _global_tokenizer = tokenizer


def compute_content_hash(text: str) -> str:
    """Computes a deterministic SHA-256 fingerprint for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_doc_id(source: str, prefix: str | None = None) -> str:
    """Generates a stable, deterministic logical document identifier based on source path/name.

    This identifier is independent of document content, so it remains constant across revisions.
    """
    source_str = str(source).strip() if source else "document"
    normalized = source_str.replace("\\", "/")
    clean_name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", normalized)
    hash_suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    base_id = f"{clean_name}_{hash_suffix}"
    if prefix:
        clean_prefix = re.sub(r"[^a-zA-Z0-9_\-.]", "_", prefix.strip())
        return f"{clean_prefix}_{base_id}"
    return base_id


def make_chunk_id(
    doc_id: str,
    section_index: int,
    chunk_index: int,
    content_hash: str | None = None,
) -> str:
    """Generates a deterministic chunk identifier.

    If content_hash is provided, it incorporates a fingerprint of the chunk content
    to identify this specific chunk version.
    """
    if content_hash:
        short_hash = content_hash[:8]
        return f"{doc_id}_{short_hash}_s{section_index}_c{chunk_index}"
    return f"{doc_id}_s{section_index}_c{chunk_index}"


class DocumentChunker:
    """Splits ExtractedDocument instances into tokenizer-bounded DocumentChunk objects.

    - Uses the project's embedding model tokenizer to count content tokens.
    - Preserves natural sentence and paragraph boundaries where practical.
    - Uses tokenizer offset mapping to preserve original characters, casing, and punctuation.
    - Maintains a configurable token overlap across adjacent chunks.
    - Preserves source metadata (filename, page numbers for PDF, headings for DOCX/Markdown).
    - Never merges across sections with distinct source metadata.
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        model_name: str | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.model_name = model_name or settings.embed_model

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = get_default_tokenizer(self.model_name)
        return self._tokenizer

    @property
    def effective_chunk_size(self) -> int:
        """Content-token limit accounting for special tokens (e.g. [CLS], [SEP])."""
        try:
            special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        except Exception:
            special_tokens = 2

        model_max = getattr(self.tokenizer, "model_max_length", None)
        if model_max is not None and 0 < model_max < 100000:
            max_allowed = model_max - special_tokens
            return min(self.chunk_size, max_allowed)
        return self.chunk_size

    def chunk_document(
        self,
        document: ExtractedDocument,
        doc_id: str | None = None,
    ) -> list[DocumentChunk]:
        """Splits an ExtractedDocument into a list of DocumentChunk instances.

        Raises:
            EmptyDocumentError: If document contains no extractable text across all sections.
        """
        non_empty_sections = [
            (idx, sec)
            for idx, sec in enumerate(document.sections)
            if sec.text and sec.text.strip()
        ]

        if not non_empty_sections:
            raise EmptyDocumentError(
                f"Document '{document.source}' contains no extractable text to chunk."
            )

        # Stable logical document identity across revisions
        logical_doc_id = (
            doc_id
            or document.metadata.get("doc_id")
            or document.metadata.get("document_id")
            or document.metadata.get("source_id")
            or make_doc_id(document.source)
        )
        all_chunks: list[DocumentChunk] = []
        global_chunk_idx = 0

        for section_idx, section in non_empty_sections:
            section_chunks, global_chunk_idx = self._chunk_section(
                section=section,
                section_idx=section_idx,
                doc_id=logical_doc_id,
                source=document.source,
                chunk_index_start=global_chunk_idx,
            )
            all_chunks.extend(section_chunks)

        return all_chunks

    def _chunk_section(
        self,
        section: ExtractedSection,
        section_idx: int,
        doc_id: str,
        source: str,
        chunk_index_start: int,
    ) -> tuple[list[DocumentChunk], int]:
        text = section.text.strip()
        if not text:
            return [], chunk_index_start

        # Tokenize section text
        encoding = self._tokenize(text)
        input_ids = encoding["input_ids"]
        offsets = encoding.get("offset_mapping")
        total_tokens = len(input_ids)

        if total_tokens == 0:
            return [], chunk_index_start

        limit = self.effective_chunk_size
        overlap = self.chunk_overlap
        min_chunk_len = max(overlap + 5, int(limit * 0.5))

        chunks: list[DocumentChunk] = []
        global_chunk_idx = chunk_index_start
        sec_chunk_idx = 0

        # Fast path: section fits entirely in one chunk
        if total_tokens <= limit:
            chunk_content_hash = compute_content_hash(text)
            chunk_id = make_chunk_id(
                doc_id, section_idx, global_chunk_idx, content_hash=chunk_content_hash
            )
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=source,
                text=text,
                token_count=total_tokens,
                section_index=section_idx,
                chunk_index=global_chunk_idx,
                page_number=section.page_number,
                heading=section.heading,
                content_hash=chunk_content_hash,
                metadata={
                    **section.metadata,
                    "section_chunk_index": 0,
                    "is_single_chunk": True,
                    "content_hash": chunk_content_hash,
                },
            )
            return [chunk], global_chunk_idx + 1

        # Multi-chunk splitting
        start_idx = 0
        while start_idx < total_tokens:
            max_end_idx = min(start_idx + limit, total_tokens)

            if max_end_idx == total_tokens:
                end_idx = total_tokens
            else:
                candidate_min = start_idx + min_chunk_len
                end_idx = self._find_break_point(text, offsets, candidate_min, max_end_idx)

            chunk_text = self._slice_text(text, offsets, input_ids, start_idx, end_idx)
            token_count = self._count_tokens(chunk_text)

            # Strict limit enforcement: trim if character slicing or boundary caused token overflow
            while token_count > limit and end_idx > start_idx + 1:
                end_idx -= 1
                chunk_text = self._slice_text(text, offsets, input_ids, start_idx, end_idx)
                token_count = self._count_tokens(chunk_text)

            chunk_content_hash = compute_content_hash(chunk_text)
            chunk_id = make_chunk_id(
                doc_id, section_idx, global_chunk_idx, content_hash=chunk_content_hash
            )
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=source,
                text=chunk_text,
                token_count=token_count,
                section_index=section_idx,
                chunk_index=global_chunk_idx,
                page_number=section.page_number,
                heading=section.heading,
                content_hash=chunk_content_hash,
                metadata={
                    **section.metadata,
                    "section_chunk_index": sec_chunk_idx,
                    "token_start": start_idx,
                    "token_end": end_idx,
                    "content_hash": chunk_content_hash,
                },
            )
            chunks.append(chunk)
            global_chunk_idx += 1
            sec_chunk_idx += 1

            if end_idx >= total_tokens:
                break

            # Calculate next start index with overlap and natural start boundary
            target_next = max(start_idx + 1, end_idx - overlap)
            next_start = self._find_start_break_point(
                text=text,
                offsets=offsets,
                target_idx=target_next,
                min_bound=start_idx + 1,
                max_bound=end_idx - 1,
            )
            start_idx = next_start

        return chunks, global_chunk_idx

    def _tokenize(self, text: str) -> dict[str, Any]:
        """Encodes text returning input_ids and offset_mapping where available."""
        try:
            enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            input_ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "__getitem__") else enc.input_ids
            offsets = enc.get("offset_mapping") if isinstance(enc, dict) else getattr(enc, "offset_mapping", None)
            return {"input_ids": list(input_ids), "offset_mapping": list(offsets) if offsets is not None else None}
        except Exception:
            # Fallback if tokenizer does not support return_offsets_mapping
            enc = self.tokenizer(text, add_special_tokens=False)
            input_ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "__getitem__") else enc.input_ids
            return {"input_ids": list(input_ids), "offset_mapping": None}

    def _count_tokens(self, text: str) -> int:
        """Counts content tokens in text without adding special tokens."""
        if not text:
            return 0
        enc = self.tokenizer(text, add_special_tokens=False)
        ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "__getitem__") else enc.input_ids
        return len(ids)

    def _slice_text(
        self,
        text: str,
        offsets: list[tuple[int, int]] | None,
        input_ids: list[int],
        start_idx: int,
        end_idx: int,
    ) -> str:
        """Slices original text using exact offsets to preserve casing, punctuation, and Unicode."""
        if offsets and len(offsets) >= end_idx and start_idx < len(offsets):
            char_start = offsets[start_idx][0]
            char_end = offsets[end_idx - 1][1]
            return text[char_start:char_end].strip()

        # Fallback when offsets are unavailable
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(input_ids[start_idx:end_idx], skip_special_tokens=True).strip()
        return text.strip()

    def _find_break_point(
        self,
        text: str,
        offsets: list[tuple[int, int]] | None,
        min_candidate_idx: int,
        max_candidate_idx: int,
    ) -> int:
        """Scans backwards from max_candidate_idx to find the most natural break point."""
        if not offsets or min_candidate_idx >= max_candidate_idx:
            return max_candidate_idx

        paragraph_breaks: list[int] = []
        sentence_breaks: list[int] = []
        line_breaks: list[int] = []
        clause_breaks: list[int] = []

        for k in range(max_candidate_idx, min_candidate_idx - 1, -1):
            if k >= len(offsets) or k < 1:
                continue

            end_char = offsets[k - 1][1]
            next_char = offsets[k][0]
            gap = text[end_char:next_char]
            prefix = text[:end_char]

            # Paragraph break (\n\n)
            if "\n\n" in gap or prefix.endswith("\n\n"):
                paragraph_breaks.append(k)
                continue

            # Sentence break (. / ! / ?)
            if prefix.endswith((".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'")):
                # Ensure it is followed by whitespace and not an internal decimal or abbreviation
                if gap and not (next_char < len(text) and text[next_char].isdigit()):
                    sentence_breaks.append(k)
                    continue

            # Line break (\n)
            if "\n" in gap or prefix.endswith("\n"):
                line_breaks.append(k)
                continue

            # Clause break (; / :)
            if prefix.endswith((";", ":")) and gap:
                clause_breaks.append(k)

        if paragraph_breaks:
            return paragraph_breaks[0]
        if sentence_breaks:
            return sentence_breaks[0]
        if line_breaks:
            return line_breaks[0]
        if clause_breaks:
            return clause_breaks[0]

        return max_candidate_idx

    def _find_start_break_point(
        self,
        text: str,
        offsets: list[tuple[int, int]] | None,
        target_idx: int,
        min_bound: int,
        max_bound: int,
    ) -> int:
        """Finds a natural start break point near target_idx within [min_bound, max_bound]."""
        if not offsets or min_bound >= max_bound:
            return max(min_bound, min(target_idx, max_bound))

        window_start = max(min_bound, target_idx - 8)
        window_end = min(max_bound, target_idx + 8)

        candidates: list[int] = []
        for k in range(window_start, window_end + 1):
            if k >= len(offsets) or k < 1:
                continue

            end_char = offsets[k - 1][1]
            next_char = offsets[k][0]
            gap = text[end_char:next_char]
            prefix = text[:end_char]

            if "\n\n" in gap or prefix.endswith("\n\n"):
                candidates.append(k)
            elif (
                prefix.endswith((".", "!", "?", '."', '!"', '?"', ".'"))
                and gap
                and not (next_char < len(text) and text[next_char].isdigit())
            ):
                candidates.append(k)

        if candidates:
            return min(candidates, key=lambda c: abs(c - target_idx))

        return max(min_bound, min(target_idx, max_bound))


def chunk_document(
    document: ExtractedDocument,
    tokenizer: Any | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    doc_id: str | None = None,
) -> list[DocumentChunk]:
    """Convenience function to chunk an ExtractedDocument using default or custom settings."""
    chunker = DocumentChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunker.chunk_document(document, doc_id=doc_id)
