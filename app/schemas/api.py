from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the knowledge copilot.")
    limit: Optional[int] = Field(5, ge=1, le=100, description="Number of chunks to retrieve.")

class SourceDetail(BaseModel):
    source: str
    text: str
    page_number: Optional[int] = None
    heading: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDetail]
    metadata: dict
