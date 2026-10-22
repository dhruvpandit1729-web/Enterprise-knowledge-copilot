from __future__ import annotations

import hashlib
import pytest
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.ingestion.embedder import (
    EmbeddingService,
    get_default_embedder,
    set_default_embedder,
)
from app.schemas.document import DocumentChunk


class MockSentenceTransformer:
    """Deterministic, offline fake model for fast testing without network or weights."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        results = []
        for s in sentences:
            h = hashlib.sha256(s.encode("utf-8")).digest()
            vec = [(h[i % len(h)] / 255.0) for i in range(self.dimension)]
            if normalize_embeddings:
                norm = sum(x * x for x in vec) ** 0.5 or 1.0
                vec = [x / norm for x in vec]
            results.append(vec)
        return results


def test_lazy_loading():
    service = EmbeddingService(model_name="dummy/model")
    assert service._model is None
    # Model is not loaded until .model is accessed


def test_embed_texts_validation(monkeypatch):
    mock_model = MockSentenceTransformer(dimension=16)
    service = EmbeddingService(model=mock_model)

    # Empty list
    with pytest.raises(ValueError, match="Cannot embed an empty list of texts"):
        service.embed_texts([])

    # Non-list
    with pytest.raises(TypeError, match="texts must be a list"):
        service.embed_texts("single text")  # type: ignore

    # Non-string inside list
    with pytest.raises(TypeError, match="must be a string"):
        service.embed_texts(["valid", 123])  # type: ignore

    # Empty string inside list
    with pytest.raises(ValueError, match="empty or contains only whitespace"):
        service.embed_texts(["valid", "   "])


def test_embed_texts_dimension_and_count():
    dim = 32
    mock_model = MockSentenceTransformer(dimension=dim)
    service = EmbeddingService(model=mock_model)

    assert service.dimension == dim

    texts = ["First document text.", "Second document text.", "Third document."]
    vectors = service.embed_texts(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert isinstance(vec, list)
        assert len(vec) == dim
        assert all(isinstance(val, float) for val in vec)
        # Vector is normalized
        norm = sum(val * val for val in vec) ** 0.5
        assert pytest.approx(norm, abs=1e-4) == 1.0


def test_embed_chunks():
    dim = 16
    mock_model = MockSentenceTransformer(dimension=dim)
    service = EmbeddingService(model=mock_model)

    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            source="doc.txt",
            text="Chunk 1 content.",
            token_count=5,
            section_index=0,
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d1",
            source="doc.txt",
            text="Chunk 2 content.",
            token_count=5,
            section_index=0,
            chunk_index=1,
        ),
    ]

    vectors = service.embed_chunks(chunks)
    assert len(vectors) == 2
    assert len(vectors[0]) == dim
    assert len(vectors[1]) == dim

    # Empty chunks list
    with pytest.raises(ValueError, match="Cannot embed an empty list of chunks"):
        service.embed_chunks([])

    # Invalid item in list
    with pytest.raises(TypeError, match="is not a DocumentChunk"):
        service.embed_chunks(["not a chunk"])  # type: ignore


def test_embed_query():
    dim = 16
    mock_model = MockSentenceTransformer(dimension=dim)
    service = EmbeddingService(model=mock_model)

    vec = service.embed_query("What is the policy?")
    assert isinstance(vec, list)
    assert len(vec) == dim

    with pytest.raises(ValueError, match="query cannot be empty or whitespace-only"):
        service.embed_query("   ")

    with pytest.raises(TypeError, match="query must be a string"):
        service.embed_query(None)  # type: ignore


def test_default_embedder_singleton():
    set_default_embedder(None)
    embedder1 = get_default_embedder()
    embedder2 = get_default_embedder()
    assert embedder1 is embedder2
    set_default_embedder(None)


def test_offline_cached_sentence_transformer_smoke():
    """Smoke test using the cached all-MiniLM-L6-v2 without network calls."""
    try:
        model = SentenceTransformer(settings.embed_model, local_files_only=True)
    except Exception:
        pytest.skip("Local SentenceTransformer model weights not cached.")

    service = EmbeddingService(model=model)
    assert service.dimension == 384

    vectors = service.embed_texts(["Policy compliance requires strict token limits."])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
