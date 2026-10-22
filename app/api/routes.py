from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from app.api.dependencies import (
    get_document_store,
    get_dispatcher,
    get_chunker,
    get_qa_service
)
from app.schemas.api import QueryRequest, QueryResponse, SourceDetail
from app.core.config import settings
from app.retrieval.store import ChromaDocumentStore
from app.ingestion.dispatcher import DocumentDispatcher
from app.ingestion.chunker import DocumentChunker
import os
from pathlib import Path

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    store: ChromaDocumentStore = Depends(get_document_store),
    dispatcher: DocumentDispatcher = Depends(get_dispatcher),
    chunker: DocumentChunker = Depends(get_chunker)
):
    # 1. Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {settings.max_upload_mb}MB"
        )

    # 2. Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in dispatcher.supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension {ext}. Supported: {', '.join(dispatcher.supported_extensions)}"
        )

    # 3. Process file
    # Write to temporary file because DocumentDispatcher expects a path
    temp_path = Path(f"temp_{file.filename}")
    try:
        if not content:
            raise ValueError("Uploaded file is empty.")

        with open(temp_path, "wb") as f:
            f.write(content)

        extracted_doc = dispatcher.parse_file(temp_path, original_filename=file.filename)
        chunks = chunker.chunk_document(extracted_doc)
        count = store.index_document_chunks(chunks)

        return {
            "status": "indexed",
            "filename": file.filename,
            "chunks_indexed": count
        }
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )
    finally:
        if temp_path.exists():
            os.remove(temp_path)

@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    qa_service = Depends(get_qa_service)
):
    try:
        # Ensure we await the async method
        answer = await qa_service.answer_question(request.question, limit=request.limit)

        sources = [
            SourceDetail(
                source=sd.source,
                text=getattr(sd, 'text', "Context omitted"),
                page_number=getattr(sd, 'page_number', None),
                heading=getattr(sd, 'heading', None)
            )
            for sd in answer.source_details
        ]

        return QueryResponse(
            answer=answer.text,
            sources=sources,
            metadata={
                "model": answer.model,
                "chunks_retrieved": len(answer.source_details)
            }
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )
