import pytest
from unittest.mock import MagicMock, AsyncMock
from app.service import QuestionAnsweringService
from app.generation.service import GeneratedAnswer
from app.schemas.document import DocumentChunk

@pytest.mark.asyncio
async def test_qa_pipeline_flow():
    mock_store = MagicMock()
    mock_generation = AsyncMock()

    chunk = DocumentChunk(
        chunk_id="1", doc_id="d1", source="doc.pdf", text="content",
        token_count=10, section_index=0, chunk_index=0
    )
    mock_store.search.return_value = [chunk]
    mock_generation.generate_answer.return_value = GeneratedAnswer(
        text="Sample answer", sources=["doc.pdf"], model="qwen2.5:7b"
    )

    service = QuestionAnsweringService(document_store=mock_store, generation_service=mock_generation)
    res = await service.answer_question("test question")

    assert res.text == "Sample answer"
    mock_store.search.assert_called_once_with("test question", limit=5)
    mock_generation.generate_answer.assert_called_once_with("test question", [chunk])

@pytest.mark.asyncio
async def test_qa_pipeline_empty_retrieval():
    mock_store = MagicMock()
    mock_generation = AsyncMock()
    mock_store.search.return_value = []

    service = QuestionAnsweringService(document_store=mock_store, generation_service=mock_generation)
    res = await service.answer_question("test question")

    assert "not provide enough information" in res.text
    mock_generation.generate_answer.assert_not_called()
