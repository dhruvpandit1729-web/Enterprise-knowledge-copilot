from __future__ import annotations

import re
from pathlib import Path
import pytest
from transformers import AutoTokenizer

from app.core.config import settings
from app.ingestion.chunker import DocumentChunker, chunk_document, make_chunk_id, make_doc_id
from app.ingestion.exceptions import EmptyDocumentError
from app.schemas.document import ExtractedDocument, ExtractedSection


class MockTokenizer:
    """Fast, deterministic, offline tokenizer for isolated unit tests."""

    def __init__(self, model_max_length: int = 512) -> None:
        self.model_max_length = model_max_length
        self._vocab: dict[str, int] = {}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = True,
        add_special_tokens: bool = False,
    ):
        pattern = re.compile(r"\w+|[^\w\s]")
        input_ids = []
        offset_mapping = []

        for match in pattern.finditer(text):
            token_str = match.group(0)
            if token_str not in self._vocab:
                self._vocab[token_str] = len(self._vocab) + 1
            input_ids.append(self._vocab[token_str])
            offset_mapping.append((match.start(), match.end()))

        class Encoding:
            def __init__(self, ids: list[int], offsets: list[tuple[int, int]] | None):
                self.input_ids = ids
                self.offset_mapping = offsets

            def __getitem__(self, item: str):
                if item == "input_ids":
                    return self.input_ids
                if item == "offset_mapping":
                    return self.offset_mapping
                raise KeyError(item)

            def get(self, item: str, default=None):
                if item == "input_ids":
                    return self.input_ids
                if item == "offset_mapping":
                    return self.offset_mapping
                return default

        return Encoding(input_ids, offset_mapping if return_offsets_mapping else None)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, "") for i in token_ids)


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


def test_short_text_single_chunk(mock_tokenizer: MockTokenizer):
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=50, chunk_overlap=10)
    section = ExtractedSection(text="This is a short policy statement.", page_number=1)
    doc = ExtractedDocument(source="short.txt", sections=[section])

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "This is a short policy statement."
    assert chunk.token_count <= 50
    assert chunk.page_number == 1
    assert chunk.section_index == 0
    assert chunk.chunk_index == 0


def test_long_text_multiple_chunks(mock_tokenizer: MockTokenizer):
    # Create text with multiple sentences exceeding chunk_size=30
    sentences = [f"This is sentence number {i} for policy testing." for i in range(1, 15)]
    full_text = " ".join(sentences)

    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=30, chunk_overlap=10)
    doc = ExtractedDocument(
        source="long.txt",
        sections=[ExtractedSection(text=full_text, page_number=None, heading="Section 1")],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1

    # Every chunk meets the token limit
    for chunk in chunks:
        assert chunk.token_count <= 30
        assert chunk.heading == "Section 1"
        assert chunk.page_number is None


def test_every_chunk_meets_token_limit_strict(mock_tokenizer: MockTokenizer):
    # Continuous text without punctuation to test token-window fallback
    unbroken_text = "word " * 200
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=40, chunk_overlap=10)
    doc = ExtractedDocument(
        source="unbroken.txt",
        sections=[ExtractedSection(text=unbroken_text)],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 40


def test_overlap_behavior(mock_tokenizer: MockTokenizer):
    sentences = [f"Item {i} is strictly required." for i in range(1, 20)]
    full_text = " ".join(sentences)

    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=30, chunk_overlap=10)
    doc = ExtractedDocument(
        source="overlap.txt",
        sections=[ExtractedSection(text=full_text)],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2

    # Verify that consecutive chunks share text content
    for i in range(len(chunks) - 1):
        c1_text = chunks[i].text
        c2_text = chunks[i + 1].text
        # Split into words and check common words
        c1_words = set(c1_text.split()[-10:])
        c2_words = set(c2_text.split()[:10])
        common = c1_words.intersection(c2_words)
        assert len(common) > 0, f"Expected overlap between chunk {i} and {i+1}"


def test_sentence_and_paragraph_boundaries(mock_tokenizer: MockTokenizer):
    p1 = "First paragraph sentence one. First paragraph sentence two."
    p2 = "Second paragraph sentence three. Second paragraph sentence four."
    full_text = f"{p1}\n\n{p2}"

    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=12, chunk_overlap=3)
    doc = ExtractedDocument(
        source="boundaries.txt",
        sections=[ExtractedSection(text=full_text)],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2

    # Chunks should not cut words in half
    for chunk in chunks:
        assert not chunk.text.endswith(("sentenc", "paragrap", "Fir", "Sec"))


def test_unicode_and_punctuation_preservation(mock_tokenizer: MockTokenizer):
    text = "Policy POL-2024-v1.2: Effective 2026-10-20. Total fee: $1,234.56. Status: active (100%)."
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=50, chunk_overlap=10)
    doc = ExtractedDocument(
        source="policy.txt",
        sections=[ExtractedSection(text=text)],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "POL-2024-v1.2" in chunks[0].text
    assert "2026-10-20" in chunks[0].text
    assert "$1,234.56" in chunks[0].text
    assert "active (100%)" in chunks[0].text


def test_empty_document_raises_empty_document_error(mock_tokenizer: MockTokenizer):
    chunker = DocumentChunker(tokenizer=mock_tokenizer)

    # Document with no sections
    doc_empty = ExtractedDocument(source="empty.txt", sections=[])
    with pytest.raises(EmptyDocumentError, match="contains no extractable text to chunk"):
        chunker.chunk_document(doc_empty)

    # Document with only whitespace sections
    doc_ws = ExtractedDocument(
        source="ws.txt",
        sections=[ExtractedSection(text="   \n\t  "), ExtractedSection(text="")],
    )
    with pytest.raises(EmptyDocumentError, match="contains no extractable text to chunk"):
        chunker.chunk_document(doc_ws)


def test_whitespace_sections_skipped(mock_tokenizer: MockTokenizer):
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=50, chunk_overlap=10)
    doc = ExtractedDocument(
        source="mixed.txt",
        sections=[
            ExtractedSection(text="   "),  # Empty
            ExtractedSection(text="Valid section content.", page_number=1),
            ExtractedSection(text="\n\t\n"),  # Empty
            ExtractedSection(text="Another valid section.", page_number=2),
        ],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].text == "Valid section content."
    assert chunks[0].page_number == 1
    assert chunks[0].section_index == 1

    assert chunks[1].text == "Another valid section."
    assert chunks[1].page_number == 2
    assert chunks[1].section_index == 3


def test_pdf_page_metadata_preserved(mock_tokenizer: MockTokenizer):
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=50, chunk_overlap=10)
    doc = ExtractedDocument(
        source="doc.pdf",
        sections=[
            ExtractedSection(text="Page 1 text here.", page_number=1),
            ExtractedSection(text="Page 2 text here.", page_number=2),
        ],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].source == "doc.pdf"
    assert chunks[1].page_number == 2
    assert chunks[1].source == "doc.pdf"


def test_heading_metadata_preserved(mock_tokenizer: MockTokenizer):
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=50, chunk_overlap=10)
    doc = ExtractedDocument(
        source="handbook.md",
        sections=[
            ExtractedSection(
                text="Security rules text.",
                heading="Overview > Security",
                page_number=None,
            ),
        ],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].heading == "Overview > Security"
    assert chunks[0].page_number is None


def test_no_invented_page_numbers_for_docx_txt_md(mock_tokenizer: MockTokenizer):
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=50, chunk_overlap=10)
    for ext in ["file.docx", "file.txt", "file.md", "file.markdown"]:
        doc = ExtractedDocument(
            source=ext,
            sections=[ExtractedSection(text="Sample content for non-PDF formats.", page_number=None)],
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].page_number is None


def test_deterministic_chunk_ids(mock_tokenizer: MockTokenizer):
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=30, chunk_overlap=10)
    text = " ".join([f"Sentence {i} for determinism." for i in range(1, 15)])
    doc = ExtractedDocument(
        source="repeat.txt",
        sections=[ExtractedSection(text=text)],
    )

    run1 = chunker.chunk_document(doc)
    run2 = chunker.chunk_document(doc)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.doc_id == c2.doc_id
        assert c1.text == c2.text
        assert c1.token_count == c2.token_count


def test_distinct_sections_not_merged(mock_tokenizer: MockTokenizer):
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=100, chunk_overlap=20)
    doc = ExtractedDocument(
        source="distinct.pdf",
        sections=[
            ExtractedSection(text="Short page 1 content.", page_number=1),
            ExtractedSection(text="Short page 2 content.", page_number=2),
        ],
    )

    chunks = chunker.chunk_document(doc)
    # Even though both sections total ~10 tokens (< 100), they MUST remain 2 separate chunks!
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].text == "Short page 1 content."
    assert chunks[1].page_number == 2
    assert chunks[1].text == "Short page 2 content."


def test_very_long_input_and_forward_progress(mock_tokenizer: MockTokenizer):
    # 1,000 words without breaks to ensure no infinite loops and steady forward progress
    long_words = [f"word{i}" for i in range(1000)]
    text = " ".join(long_words)

    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=50, chunk_overlap=10)
    doc = ExtractedDocument(source="huge.txt", sections=[ExtractedSection(text=text)])

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 10

    # Ensure every chunk made forward progress and index increases
    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == idx
        assert chunk.token_count <= 50


def test_integration_cached_all_minilm_tokenizer():
    """Integration smoke test using the cached Hugging Face AutoTokenizer for all-MiniLM-L6-v2."""
    tokenizer = AutoTokenizer.from_pretrained(settings.embed_model, local_files_only=True)
    chunker = DocumentChunker(tokenizer=tokenizer)

    # Use project settings
    assert chunker.chunk_size == settings.chunk_size  # 200
    assert chunker.chunk_overlap == settings.chunk_overlap  # 40
    assert chunker.effective_chunk_size <= 200

    sample_text = (
        "Enterprise Knowledge Copilot provides evidence-grounded answers. "
        "Each chunk must remain strictly within the configured token limit. "
        "Sentence and paragraph boundaries should be respected whenever practical."
    )
    doc = ExtractedDocument(
        source="enterprise_copilot.txt",
        sections=[ExtractedSection(text=sample_text, heading="Intro")],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].token_count <= 200
    assert chunks[0].heading == "Intro"
    assert chunks[0].page_number is None

    # Multi-chunk text with real tokenizer
    long_sentences = [
        f"This is policy statement {i} outlining compliance requirements for section 4.1."
        for i in range(1, 40)
    ]
    long_doc = ExtractedDocument(
        source="compliance.pdf",
        sections=[ExtractedSection(text=" ".join(long_sentences), page_number=1)],
    )

    multi_chunks = chunker.chunk_document(long_doc)
    assert len(multi_chunks) > 1
    for c in multi_chunks:
        assert c.token_count <= 200
        assert c.page_number == 1


def test_logical_doc_id_stability_across_content_edits(mock_tokenizer: MockTokenizer):
    """Verifies that doc_id is stable when content is updated, while content_hash changes."""
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=50, chunk_overlap=10)

    doc_v1 = ExtractedDocument(
        source="handbook.md",
        sections=[ExtractedSection(text="Original handbook rule version 1.")],
        metadata={"doc_id": "handbook_core"},
    )
    doc_v2 = ExtractedDocument(
        source="handbook.md",
        sections=[ExtractedSection(text="Completely modified handbook rules version 2.")],
        metadata={"doc_id": "handbook_core"},
    )

    chunks_v1 = chunker.chunk_document(doc_v1)
    chunks_v2 = chunker.chunk_document(doc_v2)

    # Logical doc_id is preserved
    assert chunks_v1[0].doc_id == "handbook_core"
    assert chunks_v2[0].doc_id == "handbook_core"

    # Content fingerprints detect content change
    assert chunks_v1[0].content_hash != chunks_v2[0].content_hash

    # Chunk IDs differ because chunk identity captures the content version
    assert chunks_v1[0].chunk_id != chunks_v2[0].chunk_id


def test_doc_id_disambiguation_with_same_filename(mock_tokenizer: MockTokenizer):
    """Verifies that identical filenames with different paths or upload IDs yield distinct doc_ids."""
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=50, chunk_overlap=10)

    # Different paths with same filename
    doc_hr = ExtractedDocument(
        source="hr/policy.txt",
        sections=[ExtractedSection(text="HR policy.")],
    )
    doc_it = ExtractedDocument(
        source="it/policy.txt",
        sections=[ExtractedSection(text="IT policy.")],
    )

    chunks_hr = chunker.chunk_document(doc_hr)
    chunks_it = chunker.chunk_document(doc_it)

    assert chunks_hr[0].doc_id != chunks_it[0].doc_id

    # Identical filename with explicit metadata doc_id
    doc_upload1 = ExtractedDocument(
        source="policy.txt",
        sections=[ExtractedSection(text="Upload 1.")],
        metadata={"doc_id": "upload_uuid_001"},
    )
    doc_upload2 = ExtractedDocument(
        source="policy.txt",
        sections=[ExtractedSection(text="Upload 2.")],
        metadata={"doc_id": "upload_uuid_002"},
    )

    c1 = chunker.chunk_document(doc_upload1)
    c2 = chunker.chunk_document(doc_upload2)

    assert c1[0].doc_id == "upload_uuid_001"
    assert c2[0].doc_id == "upload_uuid_002"
    assert c1[0].doc_id != c2[0].doc_id


def test_content_hash_and_chunk_id_determinism(mock_tokenizer: MockTokenizer):
    """Verifies that chunk_id and content_hash are completely deterministic."""
    chunker = DocumentChunker(tokenizer=mock_tokenizer, chunk_size=30, chunk_overlap=10)
    doc = ExtractedDocument(
        source="deterministic.txt",
        sections=[ExtractedSection(text="Strictly repeatable content testing.")],
    )

    c_run1 = chunker.chunk_document(doc)
    c_run2 = chunker.chunk_document(doc)

    assert c_run1[0].chunk_id == c_run2[0].chunk_id
    assert c_run1[0].doc_id == c_run2[0].doc_id
    assert c_run1[0].content_hash == c_run2[0].content_hash
    assert c_run1[0].content_hash is not None
