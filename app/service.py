from __future__ import annotations

from app.generation.service import GenerationService, GeneratedAnswer
from app.retrieval.store import ChromaDocumentStore, get_default_store

class QuestionAnsweringService:
    def __init__(
        self,
        document_store: ChromaDocumentStore | None = None,
        generation_service: GenerationService | None = None,
    ):
        self.document_store = document_store or get_default_store()
        self.generation_service = generation_service

    async def answer_question(self, question: str, limit: int = 5) -> GeneratedAnswer:
        chunks = self.document_store.search(question, limit=limit)
        if not chunks:
            return GeneratedAnswer(
                text="The available documents do not provide enough information.",
                sources=[],
                model="none",
            )
        return await self.generation_service.generate_answer(question, chunks)
