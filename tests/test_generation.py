import pytest
from unittest.mock import AsyncMock
from app.generation.service import GenerationService
from app.schemas.document import DocumentChunk

@pytest.mark.asyncio
async def test_generate_answer():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "The answer is based on documents."
    service = GenerationService(llm_client=mock_llm)

    chunk = DocumentChunk(
        chunk_id="1", doc_id="d1", source="doc.pdf", text="content",
        token_count=10, section_index=0, chunk_index=0
    )

    result = await service.generate_answer("What is the content?", [chunk])

    assert "answer" in result.text.lower()
    assert "doc.pdf" in result.sources
    mock_llm.generate.assert_called_once()
