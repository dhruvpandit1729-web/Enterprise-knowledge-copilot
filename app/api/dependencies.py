from fastapi import Depends
from app.retrieval.store import ChromaDocumentStore, get_default_store
from app.ingestion.dispatcher import DocumentDispatcher
from app.ingestion.chunker import DocumentChunker
from app.service import QuestionAnsweringService
from app.generation.service import GenerationService
from app.generation.client import OllamaClient
from app.core.config import settings

# Singletons for the app lifetime
_doc_store = get_default_store()
_gen_service = GenerationService(llm_client=OllamaClient())

def get_document_store() -> ChromaDocumentStore:
    return _doc_store

def get_dispatcher() -> DocumentDispatcher:
    return DocumentDispatcher()

def get_chunker() -> DocumentChunker:
    return DocumentChunker()

def get_qa_service() -> QuestionAnsweringService:
    return QuestionAnsweringService(
        document_store=_doc_store,
        generation_service=_gen_service
    )
