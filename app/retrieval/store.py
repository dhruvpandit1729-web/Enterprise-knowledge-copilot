from __future__ import annotations

from pathlib import Path
from typing import Any
import chromadb

from app.core.config import settings
from app.ingestion.embedder import EmbeddingService, get_default_embedder
from app.schemas.document import DocumentChunk

_global_store: Any | None = None


def get_default_store(
    persist_path: str | Path | None = None,
    collection_name: str = "documents",
    embedding_service: EmbeddingService | None = None,
) -> ChromaDocumentStore:
    """Lazily loads and reuses the default ChromaDocumentStore instance."""
    global _global_store
    if _global_store is None:
        _global_store = ChromaDocumentStore(
            persist_path=persist_path,
            collection_name=collection_name,
            embedding_service=embedding_service,
        )
    return _global_store


def set_default_store(store: Any | None) -> None:
    """Sets or resets the default ChromaDocumentStore instance (useful for test isolation)."""
    global _global_store
    _global_store = store


class ChromaDocumentStore:
    """Persistent ChromaDB vector store for DocumentChunk indexing and management.

    - Uses chromadb.PersistentClient with the configured project-relative CHROMA_PATH.
    - Explicitly creates and validates the collection with cosine distance (hnsw:space = cosine).
    - Uses an explicit embedding service rather than letting ChromaDB load a second implicit model.
    - Stores chunk_id as record ID, text as document, and scalar fields as metadata.
    - Omits optional metadata fields when None (e.g. page_number for non-PDFs).
    - Safely replaces document chunks: generates embeddings and upserts new chunks before deleting obsolete IDs.
    - Provides safe document-level deletion without affecting unrelated documents.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        collection_name: str = "documents",
        embedding_service: EmbeddingService | Any | None = None,
        client: Any | None = None,
    ) -> None:
        self.persist_path = Path(persist_path or settings.chroma_path).resolve()
        self.collection_name = collection_name
        self._embedding_service = embedding_service
        self._client = client
        self._collection: Any | None = None

    @property
    def embedding_service(self) -> Any:
        if self._embedding_service is None:
            self._embedding_service = get_default_embedder()
        return self._embedding_service

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_path))
        return self._client

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self._get_or_create_collection()
        return self._collection

    def _get_or_create_collection(self) -> Any:
        """Retrieves or creates the collection, verifying cosine distance configuration."""
        existing_collections = self.client.list_collections()
        existing_names = [
            c.name if hasattr(c, "name") else str(c) for c in existing_collections
        ]

        if self.collection_name in existing_names:
            col = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None,
            )
            meta = col.metadata or {}
            space = meta.get("hnsw:space")
            if space != "cosine":
                raise ValueError(
                    f"Collection '{self.collection_name}' has distance metric '{space}', "
                    f"but 'cosine' is required by project configuration."
                )
            return col

        return self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def index_document_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Indexes or safely replaces all chunks belonging to one or more documents.

        Process:
        1. Validates input chunks.
        2. Groups chunks by doc_id.
        3. For each doc_id:
           - Retrieves existing chunk IDs in ChromaDB for that doc_id.
           - Generates embeddings via the embedding service.
           - Upserts new chunks to ChromaDB.
           - Deletes any obsolete chunk IDs left over from earlier versions.

        Raises:
            TypeError: If chunks is not a list or contains non-DocumentChunk items.
            ValueError: If chunks is empty.
        """
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be a list, got {type(chunks).__name__}.")
        if len(chunks) == 0:
            raise ValueError("Cannot index an empty list of document chunks.")

        chunks_by_doc: dict[str, list[DocumentChunk]] = {}
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                raise TypeError(
                    f"Item at index {idx} is not a DocumentChunk (got {type(chunk).__name__})."
                )
            if not chunk.chunk_id or not chunk.doc_id or not chunk.text.strip():
                raise ValueError(
                    f"Chunk at index {idx} has missing chunk_id, doc_id, or text."
                )
            chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)

        total_indexed = 0

        for doc_id, doc_chunks in chunks_by_doc.items():
            # 1. Query existing chunk IDs for this document
            existing_result = self.collection.get(where={"doc_id": doc_id})
            existing_ids = set(existing_result.get("ids", []))

            # 2. Compute embeddings (if this fails, ChromaDB is untouched)
            embeddings = self.embedding_service.embed_chunks(doc_chunks)

            # 3. Prepare ChromaDB payloads
            new_ids = [c.chunk_id for c in doc_chunks]
            new_documents = [c.text for c in doc_chunks]
            new_metadatas = [self._chunk_to_metadata(c) for c in doc_chunks]

            # 4. Upsert new chunks
            self.collection.upsert(
                ids=new_ids,
                embeddings=embeddings,
                documents=new_documents,
                metadatas=new_metadatas,
            )

            # 5. Delete obsolete chunk IDs belonging to this doc_id
            obsolete_ids = existing_ids - set(new_ids)
            if obsolete_ids:
                self.collection.delete(ids=list(obsolete_ids))

            total_indexed += len(doc_chunks)

        return total_indexed

    def delete_document(self, doc_id: str) -> int:
        """Safely deletes all chunks associated with doc_id.

        Does not affect other documents in the store.
        Returns the number of deleted chunks.

        Raises:
            TypeError: If doc_id is not a string.
            ValueError: If doc_id is empty or whitespace-only.
        """
        if not isinstance(doc_id, str):
            raise TypeError(f"doc_id must be a string, got {type(doc_id).__name__}.")
        if not doc_id.strip():
            raise ValueError("doc_id cannot be empty or whitespace-only.")

        existing_result = self.collection.get(where={"doc_id": doc_id})
        existing_ids = existing_result.get("ids", [])
        if existing_ids:
            self.collection.delete(ids=existing_ids)
            return len(existing_ids)
        return 0

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieves a single chunk by chunk_id, or None if not found."""
        if not chunk_id or not isinstance(chunk_id, str):
            return None

        result = self.collection.get(ids=[chunk_id])
        ids = result.get("ids", [])
        if not ids:
            return None

        doc = result["documents"][0]
        meta = result["metadatas"][0] if result.get("metadatas") else {}
        return self._record_to_chunk(chunk_id=chunk_id, text=doc, metadata=meta)

    def get_document_chunks(self, doc_id: str) -> list[DocumentChunk]:
        """Retrieves all chunks associated with a doc_id, sorted by chunk_index."""
        if not doc_id or not isinstance(doc_id, str):
            return []

        result = self.collection.get(where={"doc_id": doc_id})
        ids = result.get("ids", [])
        if not ids:
            return []

        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        chunks: list[DocumentChunk] = []
        for i, chunk_id in enumerate(ids):
            text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            chunks.append(self._record_to_chunk(chunk_id=chunk_id, text=text, metadata=meta))

        chunks.sort(key=lambda c: (c.section_index, c.chunk_index))
        return chunks

    def count(self) -> int:
        """Returns the total number of chunks stored in the collection."""
        return int(self.collection.count())

    def search(self, query: str, limit: int = 5) -> list[DocumentChunk]:
        """Performs semantic search, returning the top N chunks."""
        query_embedding = self.embedding_service.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        chunks: list[DocumentChunk] = []
        for i, chunk_id in enumerate(ids):
            text = documents[i]
            meta = metadatas[i] or {}
            chunks.append(self._record_to_chunk(chunk_id=chunk_id, text=text, metadata=meta))
        return chunks

    def get_all_doc_ids(self) -> list[str]:
        """Returns a list of all unique doc_ids stored in the collection."""
        result = self.collection.get()
        metas = result.get("metadatas", [])
        unique_docs = sorted({m.get("doc_id") for m in metas if m and "doc_id" in m})
        return [str(d) for d in unique_docs]

    @staticmethod
    def _chunk_to_metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
        """Converts DocumentChunk fields to ChromaDB-compatible metadata (scalars only)."""
        meta: dict[str, str | int | float | bool] = {
            "doc_id": chunk.doc_id,
            "source": chunk.source,
            "section_index": chunk.section_index,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
        }

        # Include page_number, heading, and content_hash ONLY when genuinely present (omit None)
        if chunk.page_number is not None:
            meta["page_number"] = int(chunk.page_number)
        if chunk.heading is not None:
            meta["heading"] = str(chunk.heading)
        if chunk.content_hash is not None:
            meta["content_hash"] = str(chunk.content_hash)

        # Include additional scalar fields from chunk.metadata
        for k, v in chunk.metadata.items():
            if k in meta or v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v

        return meta

    @staticmethod
    def _record_to_chunk(
        chunk_id: str,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> DocumentChunk:
        """Reconstructs a DocumentChunk from a ChromaDB record."""
        meta = metadata or {}
        reserved_keys = {
            "doc_id",
            "source",
            "section_index",
            "chunk_index",
            "token_count",
            "page_number",
            "heading",
            "content_hash",
        }
        extra_metadata = {k: v for k, v in meta.items() if k not in reserved_keys}

        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=str(meta.get("doc_id", "")),
            source=str(meta.get("source", "")),
            text=text,
            token_count=int(meta.get("token_count", 0)),
            section_index=int(meta.get("section_index", 0)),
            chunk_index=int(meta.get("chunk_index", 0)),
            page_number=meta.get("page_number"),
            heading=meta.get("heading"),
            content_hash=meta.get("content_hash"),
            metadata=extra_metadata,
        )
