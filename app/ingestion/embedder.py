from __future__ import annotations

from typing import Any
import numpy as np

from app.core.config import settings
from app.schemas.document import DocumentChunk

_global_embedder: Any | None = None


def get_default_embedder(model_name: str | None = None) -> EmbeddingService:
    """Lazily loads and reuses the default EmbeddingService instance."""
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = EmbeddingService(model_name=model_name)
    return _global_embedder


def set_default_embedder(embedder: Any | None) -> None:
    """Sets or resets the default embedder instance (useful for test isolation)."""
    global _global_embedder
    _global_embedder = embedder


class EmbeddingService:
    """Modular embedding service using SentenceTransformers with lazy model loading."""

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name or settings.embed_model
        self._model = model
        self.device = device
        self.batch_size = batch_size

    @property
    def model(self) -> Any:
        """Lazily loads and returns the SentenceTransformer model instance."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimension."""
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
            if dim is not None:
                return int(dim)
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            dim = self.model.get_sentence_embedding_dimension()
            if dim is not None:
                return int(dim)
        sample = self.embed_query("probe")
        return len(sample)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generates normalized vector embeddings for a list of texts.

        Raises:
            TypeError: If texts is not a list or contains non-string items.
            ValueError: If texts is empty or contains empty/whitespace-only items.
            RuntimeError: If the output vector count does not match the input count.
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list of strings, got {type(texts).__name__}.")
        if len(texts) == 0:
            raise ValueError("Cannot embed an empty list of texts.")

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Text at index {idx} must be a string, got {type(text).__name__}."
                )
            if not text.strip():
                raise ValueError(
                    f"Text at index {idx} is empty or contains only whitespace."
                )

        raw_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Convert to list of list of floats
        if isinstance(raw_embeddings, np.ndarray):
            vectors: list[list[float]] = raw_embeddings.tolist()
        elif hasattr(raw_embeddings, "tolist"):
            vectors = raw_embeddings.tolist()
        else:
            vectors = [list(map(float, vec)) for vec in raw_embeddings]

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Generated {len(vectors)} embeddings for {len(texts)} input texts."
            )

        return vectors

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """Extracts text from a list of DocumentChunk instances and generates embeddings.

        Raises:
            TypeError: If chunks is not a list or contains non-DocumentChunk items.
            ValueError: If chunks is empty.
        """
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be a list, got {type(chunks).__name__}.")
        if len(chunks) == 0:
            raise ValueError("Cannot embed an empty list of chunks.")

        texts: list[str] = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                raise TypeError(
                    f"Item at index {idx} is not a DocumentChunk (got {type(chunk).__name__})."
                )
            texts.append(chunk.text)

        return self.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        """Generates an embedding vector for a single query string.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or whitespace-only.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")
        if not query.strip():
            raise ValueError("query cannot be empty or whitespace-only.")

        return self.embed_texts([query])[0]
