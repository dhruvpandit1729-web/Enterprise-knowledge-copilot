import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.api.dependencies import (
    get_document_store,
    get_dispatcher,
    get_chunker,
    get_qa_service
)
from app.retrieval.store import ChromaDocumentStore
from app.ingestion.dispatcher import DocumentDispatcher
from app.ingestion.chunker import DocumentChunker
from app.service import QuestionAnsweringService
from app.schemas.document import DocumentChunk

# Use an isolated store for all API tests
TEST_CHROMA_PATH = Path("temp_api_test_db")

@pytest.fixture
def client():
    # Setup isolated store for the duration of the test
    TEST_CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    store = ChromaDocumentStore(persist_path=TEST_CHROMA_PATH, collection_name="api_test_collection")
    try:
        store.client.delete_collection(name="api_test_collection")
    except:
        pass

    # Override dependencies to use isolated store and mocks
    app.dependency_overrides[get_document_store] = lambda: store
    # Important: the mock must be an instance that is returned by the dependency
    mock_qa_inst = MagicMock(spec=QuestionAnsweringService)
    app.dependency_overrides[get_qa_service] = lambda: mock_qa_inst

    with TestClient(app) as c:
        yield c

    # Cleanup
    app.dependency_overrides.clear()
    import shutil
    try:
        shutil.rmtree(TEST_CHROMA_PATH)
    except:
        pass

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_upload_document_success(client):
    # Create a dummy valid file
    filename = "test.txt"
    content = b"This is a test document for API indexing."

    # Use a real dispatcher/chunker for the upload test to verify the pipeline
    # But we use the isolated store
    files = {"file": (filename, content, "text/plain")}
    response = client.post("/documents", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "indexed"
    assert data["filename"] == filename
    assert data["chunks_indexed"] > 0

def test_upload_document_invalid_extension(client):
    filename = "test.exe"
    content = b"malicious content"
    files = {"file": (filename, content, "application/octet-stream")}
    response = client.post("/documents", files=files)

    assert response.status_code == 415
    assert "Unsupported file extension" in response.json()["detail"]

def test_upload_document_empty_file(client):
    filename = "empty.txt"
    content = b""
    files = {"file": (filename, content, "text/plain")}
    response = client.post("/documents", files=files)

    # This will likely fail at the index_document_chunks level
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]

def test_query_success(client):
    # Get the mock instance that was injected
    mock_qa = app.dependency_overrides[get_qa_service]()

    # Create a fake GeneratedAnswer instance
    class FakeAnswer:
        def __init__(self):
            self.text = "The capital of France is Paris."
            self.sources = ["france.pdf"]
            self.source_details = [
                MagicMock(source="france.pdf", text="Paris is the capital", page_number=1, heading="Geography")
            ]
            self.model = "gpt-4"

    # Correctly set the return value to an instance
    mock_qa.answer_question = AsyncMock(return_value=FakeAnswer())

    response = client.post("/query", json={"question": "What is the capital of France?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The capital of France is Paris."
    assert data["sources"][0]["source"] == "france.pdf"
    assert data["sources"][0]["page_number"] == 1
    assert data["metadata"]["model"] == "gpt-4"

def test_query_invalid_request(client):
    # Missing question
    response = client.post("/query", json={})
    assert response.status_code == 422 # Pydantic validation error

def test_query_insufficient_evidence(client):
    # Get the mock instance that was injected
    mock_qa = app.dependency_overrides[get_qa_service]()

    class AbstentionAnswer:
        def __init__(self):
            self.text = "The available documents do not provide enough information."
            self.sources = []
            self.source_details = []
            self.model = "none"

    # Correctly set the return value to an instance
    fake_abstention = AbstentionAnswer()
    mock_qa.answer_question = AsyncMock(return_value=fake_abstention)

    response = client.post("/query", json={"question": "What is the secret code?"})

    assert response.status_code == 200
    data = response.json()
    assert "do not provide enough information" in data["answer"]
    assert len(data["sources"]) == 0

def test_upload_filename_preservation(client):
    # Test that the original filename is preserved in the indexed chunks
    filename = "original_test_doc.txt"
    content = b"This is a test for filename preservation."
    files = {"file": (filename, content, "text/plain")}

    # Upload the file
    response = client.post("/documents", files=files)
    assert response.status_code == 200

    # Query for the content and verify the source metadata
    store = app.dependency_overrides[get_document_store]()
    results = store.search("filename preservation", limit=1)

    assert len(results) > 0
    assert results[0].source == filename, f"Expected source {filename}, got {results[0].source}"
