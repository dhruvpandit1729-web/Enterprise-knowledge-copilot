from __future__ import annotations

import hashlib
from pathlib import Path
import pytest
import chromadb

from app.retrieval.store import (
    ChromaDocumentStore,
    get_default_store,
    set_default_store,
)
from app.schemas.document import DocumentChunk


class FakeEmbeddingService:
    """Deterministic, offline embedding service for isolated store testing."""

    def __init__(self, dimension: int = 16, fail_on_purpose: bool = False) -> None:
        self.dimension = dimension
        self.fail_on_purpose = fail_on_purpose

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.fail_on_purpose:
            raise RuntimeError("Simulated embedding service failure.")
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("Invalid texts.")
        results = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(h[i % len(h)] / 255.0) for i in range(self.dimension)]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            results.append([round(x / norm, 6) for x in vec])
        return results

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        if self.fail_on_purpose:
            raise RuntimeError("Simulated embedding service failure.")
        if not chunks:
            raise ValueError("Empty chunks.")
        return self.embed_texts([c.text for c in chunks])

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


@pytest.fixture
def fake_embedder() -> FakeEmbeddingService:
    return FakeEmbeddingService(dimension=16)


def test_collection_creation_cosine_distance(tmp_path: Path, fake_embedder: FakeEmbeddingService):
    store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="test_col",
        embedding_service=fake_embedder,
    )

    col = store.collection
    assert col.name == "test_col"
    assert col.metadata.get("hnsw:space") == "cosine"


def test_conflicting_distance_metric_rejected(tmp_path: Path, fake_embedder: FakeEmbeddingService):
    chroma_dir = tmp_path / "chroma_conflict"
    raw_client = chromadb.PersistentClient(path=str(chroma_dir))
    # Create collection with L2 distance instead of cosine
    raw_client.create_collection(name="conflict_col", metadata={"hnsw:space": "l2"})

    store = ChromaDocumentStore(
        persist_path=chroma_dir,
        collection_name="conflict_col",
        embedding_service=fake_embedder,
    )

    with pytest.raises(ValueError, match="has distance metric 'l2', but 'cosine' is required"):
        _ = store.collection


def test_indexing_preserves_text_and_source_metadata(tmp_path: Path, fake_embedder: FakeEmbeddingService):
    store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="docs",
        embedding_service=fake_embedder,
    )

    chunk = DocumentChunk(
        chunk_id="policy.pdf_123_s0_c0",
        doc_id="policy.pdf_123",
        source="policy.pdf",
        text="Full policy text on security authentication.",
        token_count=12,
        section_index=0,
        chunk_index=0,
        page_number=1,
        heading="Security Policy",
        metadata={"custom_flag": True, "reviewer": "Alice"},
    )

    indexed_count = store.index_document_chunks([chunk])
    assert indexed_count == 1
    assert store.count() == 1

    retrieved = store.get_chunk("policy.pdf_123_s0_c0")
    assert retrieved is not None
    assert retrieved.chunk_id == "policy.pdf_123_s0_c0"
    assert retrieved.doc_id == "policy.pdf_123"
    assert retrieved.source == "policy.pdf"
    assert retrieved.text == "Full policy text on security authentication."
    assert retrieved.token_count == 12
    assert retrieved.section_index == 0
    assert retrieved.chunk_index == 0
    assert retrieved.page_number == 1
    assert retrieved.heading == "Security Policy"
    assert retrieved.metadata.get("custom_flag") is True
    assert retrieved.metadata.get("reviewer") == "Alice"


def test_optional_metadata_omitted_when_none(tmp_path: Path, fake_embedder: FakeEmbeddingService):
    store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="docs",
        embedding_service=fake_embedder,
    )

    # DOCX/TXT chunk where page_number is None and heading is None
    chunk = DocumentChunk(
        chunk_id="notes.txt_456_s0_c0",
        doc_id="notes.txt_456",
        source="notes.txt",
        text="Plain text notes without page numbers.",
        token_count=8,
        section_index=0,
        chunk_index=0,
        page_number=None,
        heading=None,
    )

    store.index_document_chunks([chunk])

    # Inspect raw metadata stored in ChromaDB
    raw_res = store.collection.get(ids=["notes.txt_456_s0_c0"])
    raw_meta = raw_res["metadatas"][0]

    assert "page_number" not in raw_meta
    assert "heading" not in raw_meta

    # Reconstructed chunk has None for both
    retrieved = store.get_chunk("notes.txt_456_s0_c0")
    assert retrieved is not None
    assert retrieved.page_number is None
    assert retrieved.heading is None


def test_persistence_survives_closing_and_reopening(tmp_path: Path, fake_embedder: FakeEmbeddingService):
    chroma_dir = tmp_path / "persist_test"

    # 1. Open store, write chunks, and let client close
    store1 = ChromaDocumentStore(
        persist_path=chroma_dir,
        collection_name="persistent_docs",
        embedding_service=fake_embedder,
    )
    chunk = DocumentChunk(
        chunk_id="chunk_persist",
        doc_id="doc_persist",
        source="data.md",
        text="Data that must persist across reboots.",
        token_count=10,
        section_index=0,
        chunk_index=0,
    )
    store1.index_document_chunks([chunk])
    assert store1.count() == 1

    # 2. Open brand new store instance on same path
    store2 = ChromaDocumentStore(
        persist_path=chroma_dir,
        collection_name="persistent_docs",
        embedding_service=fake_embedder,
    )
    assert store2.count() == 1
    reloaded = store2.get_chunk("chunk_persist")
    assert reloaded is not None
    assert reloaded.text == "Data that must persist across reboots."
    assert reloaded.doc_id == "doc_persist"


def test_reindexing_with_fewer_chunks_removes_obsolete_chunks(
    tmp_path: Path, fake_embedder: FakeEmbeddingService
):
    store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="docs",
        embedding_service=fake_embedder,
    )

    # Initial version has 3 chunks
    v1_chunks = [
        DocumentChunk(
            chunk_id=f"doc1_s0_c{i}",
            doc_id="doc1",
            source="doc1.txt",
            text=f"Paragraph {i} content.",
            token_count=5,
            section_index=0,
            chunk_index=i,
        )
        for i in range(3)
    ]
    store.index_document_chunks(v1_chunks)
    assert store.count() == 3
    assert len(store.get_document_chunks("doc1")) == 3

    # New version has only 1 chunk (e.g. document was trimmed)
    v2_chunks = [
        DocumentChunk(
            chunk_id="doc1_s0_c0",
            doc_id="doc1",
            source="doc1.txt",
            text="Updated consolidated single paragraph.",
            token_count=8,
            section_index=0,
            chunk_index=0,
        )
    ]
    store.index_document_chunks(v2_chunks)

    # Obsolete chunks (c1, c2) must have been removed
    assert store.count() == 1
    remaining = store.get_document_chunks("doc1")
    assert len(remaining) == 1
    assert remaining[0].chunk_id == "doc1_s0_c0"
    assert remaining[0].text == "Updated consolidated single paragraph."
    assert store.get_chunk("doc1_s0_c1") is None
    assert store.get_chunk("doc1_s0_c2") is None


def test_failed_embedding_preserves_previous_index(
    tmp_path: Path, fake_embedder: FakeEmbeddingService
):
    store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="docs",
        embedding_service=fake_embedder,
    )

    initial_chunks = [
        DocumentChunk(
            chunk_id=f"safe_doc_c{i}",
            doc_id="safe_doc",
            source="safe.txt",
            text=f"Original text {i}",
            token_count=5,
            section_index=0,
            chunk_index=i,
        )
        for i in range(2)
    ]
    store.index_document_chunks(initial_chunks)
    assert store.count() == 2

    # Attempt re-indexing with a failing embedder
    failing_embedder = FakeEmbeddingService(dimension=16, fail_on_purpose=True)
    failing_store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="docs",
        embedding_service=failing_embedder,
    )

    new_chunks = [
        DocumentChunk(
            chunk_id="safe_doc_c0",
            doc_id="safe_doc",
            source="safe.txt",
            text="Attempted replacement text",
            token_count=6,
            section_index=0,
            chunk_index=0,
        )
    ]

    with pytest.raises(RuntimeError, match="Simulated embedding service failure"):
        failing_store.index_document_chunks(new_chunks)

    # Previous chunks must still be intact!
    assert store.count() == 2
    chunks_after = store.get_document_chunks("safe_doc")
    assert len(chunks_after) == 2
    assert chunks_after[0].text == "Original text 0"
    assert chunks_after[1].text == "Original text 1"


def test_delete_document_preserves_unrelated_documents(
    tmp_path: Path, fake_embedder: FakeEmbeddingService
):
    store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="docs",
        embedding_service=fake_embedder,
    )

    doc_a_chunks = [
        DocumentChunk(
            chunk_id=f"doc_a_c{i}",
            doc_id="doc_a",
            source="a.txt",
            text=f"Doc A content {i}",
            token_count=5,
            section_index=0,
            chunk_index=i,
        )
        for i in range(2)
    ]
    doc_b_chunks = [
        DocumentChunk(
            chunk_id=f"doc_b_c{i}",
            doc_id="doc_b",
            source="b.txt",
            text=f"Doc B content {i}",
            token_count=5,
            section_index=0,
            chunk_index=i,
        )
        for i in range(3)
    ]

    store.index_document_chunks(doc_a_chunks)
    store.index_document_chunks(doc_b_chunks)
    assert store.count() == 5

    # Delete doc_a only
    deleted = store.delete_document("doc_a")
    assert deleted == 2

    # doc_a chunks gone
    assert len(store.get_document_chunks("doc_a")) == 0

    # doc_b chunks fully preserved
    assert len(store.get_document_chunks("doc_b")) == 3
    assert store.count() == 3

    # Delete non-existent document returns 0
    assert store.delete_document("non_existent") == 0


def test_empty_indexing_input_rejected_safely(
    tmp_path: Path, fake_embedder: FakeEmbeddingService
):
    store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="docs",
        embedding_service=fake_embedder,
    )

    chunk = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        source="d1.txt",
        text="Existing chunk.",
        token_count=5,
        section_index=0,
        chunk_index=0,
    )
    store.index_document_chunks([chunk])
    assert store.count() == 1

    # Empty list
    with pytest.raises(ValueError, match="Cannot index an empty list"):
        store.index_document_chunks([])

    # Non-list
    with pytest.raises(TypeError, match="chunks must be a list"):
        store.index_document_chunks("invalid")  # type: ignore

    # Non-DocumentChunk
    with pytest.raises(TypeError, match="is not a DocumentChunk"):
        store.index_document_chunks([chunk, "string"])  # type: ignore

    # None or whitespace text
    bad_chunk = DocumentChunk(
        chunk_id="c2",
        doc_id="d2",
        source="d2.txt",
        text="   ",
        token_count=0,
        section_index=0,
        chunk_index=0,
    )
    with pytest.raises(ValueError, match="missing chunk_id, doc_id, or text"):
        store.index_document_chunks([bad_chunk])

    # Existing records remain completely untouched
    assert store.count() == 1
    assert store.get_chunk("c1") is not None


def test_document_isolation(tmp_path: Path, fake_embedder: FakeEmbeddingService):
    store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="docs",
        embedding_service=fake_embedder,
    )

    d1 = [
        DocumentChunk(
            chunk_id="d1_c0",
            doc_id="doc_alpha",
            source="alpha.pdf",
            text="Alpha policy details.",
            token_count=5,
            section_index=0,
            chunk_index=0,
        )
    ]
    d2 = [
        DocumentChunk(
            chunk_id="d2_c0",
            doc_id="doc_beta",
            source="beta.pdf",
            text="Beta guidelines.",
            token_count=5,
            section_index=0,
            chunk_index=0,
        )
    ]

    store.index_document_chunks(d1)
    store.index_document_chunks(d2)

    assert set(store.get_all_doc_ids()) == {"doc_alpha", "doc_beta"}

    alpha_chunks = store.get_document_chunks("doc_alpha")
    assert len(alpha_chunks) == 1
    assert alpha_chunks[0].chunk_id == "d1_c0"

    beta_chunks = store.get_document_chunks("doc_beta")
    assert len(beta_chunks) == 1
    assert beta_chunks[0].chunk_id == "d2_c0"


def test_updating_logical_document_replaces_all_stale_chunks(
    tmp_path: Path, fake_embedder: FakeEmbeddingService
):
    """Verifies that editing document content replaces previous chunks without orphans."""
    store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="docs",
        embedding_service=fake_embedder,
    )

    doc_id = "employee_handbook"

    # Version 1: 3 chunks with version 1 content
    v1_chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}_hashV1_s0_c{i}",
            doc_id=doc_id,
            source="handbook.pdf",
            text=f"Version 1 rule paragraph {i}",
            token_count=6,
            section_index=0,
            chunk_index=i,
            content_hash=f"hashV1_{i}",
        )
        for i in range(3)
    ]

    store.index_document_chunks(v1_chunks)
    assert store.count() == 3

    # Version 2: edited down to 2 chunks with new content and new chunk IDs
    v2_chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}_hashV2_s0_c{i}",
            doc_id=doc_id,
            source="handbook.pdf",
            text=f"Version 2 updated rule {i}",
            token_count=6,
            section_index=0,
            chunk_index=i,
            content_hash=f"hashV2_{i}",
        )
        for i in range(2)
    ]

    store.index_document_chunks(v2_chunks)

    # Must contain exactly the 2 new chunks; all 3 V1 chunks must be gone
    assert store.count() == 2
    reloaded = store.get_document_chunks(doc_id)
    assert len(reloaded) == 2
    assert {c.chunk_id for c in reloaded} == {
        f"{doc_id}_hashV2_s0_c0",
        f"{doc_id}_hashV2_s0_c1",
    }
    assert store.get_chunk(f"{doc_id}_hashV1_s0_c0") is None
    assert store.get_chunk(f"{doc_id}_hashV1_s0_c1") is None
    assert store.get_chunk(f"{doc_id}_hashV1_s0_c2") is None


def test_distinct_logical_identities_with_same_filename(
    tmp_path: Path, fake_embedder: FakeEmbeddingService
):
    """Verifies documents sharing a filename with distinct logical IDs remain isolated."""
    store = ChromaDocumentStore(
        persist_path=tmp_path / "chroma",
        collection_name="docs",
        embedding_service=fake_embedder,
    )

    # Two separate uploads with same filename "sop.pdf" but distinct logical IDs
    hr_chunk = DocumentChunk(
        chunk_id="hr_sop_c0",
        doc_id="hr_department_sop",
        source="sop.pdf",
        text="HR Standard Operating Procedure.",
        token_count=5,
        section_index=0,
        chunk_index=0,
    )
    it_chunk = DocumentChunk(
        chunk_id="it_sop_c0",
        doc_id="it_department_sop",
        source="sop.pdf",
        text="IT Standard Operating Procedure.",
        token_count=5,
        section_index=0,
        chunk_index=0,
    )

    store.index_document_chunks([hr_chunk])
    store.index_document_chunks([it_chunk])
    assert store.count() == 2

    # Update HR SOP
    hr_updated = DocumentChunk(
        chunk_id="hr_sop_v2_c0",
        doc_id="hr_department_sop",
        source="sop.pdf",
        text="HR Standard Operating Procedure Revision 2.",
        token_count=6,
        section_index=0,
        chunk_index=0,
    )
    store.index_document_chunks([hr_updated])

    # Total remains 2: HR was replaced, IT was untouched
    assert store.count() == 2
    it_loaded = store.get_document_chunks("it_department_sop")
    assert len(it_loaded) == 1
    assert it_loaded[0].text == "IT Standard Operating Procedure."

    # Delete HR SOP
    deleted = store.delete_document("hr_department_sop")
    assert deleted == 1
    assert store.count() == 1
    assert len(store.get_document_chunks("it_department_sop")) == 1

