from __future__ import annotations
from typing import Protocol, runtime_checkable, List, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.document import DocumentChunk
import collections

class SourceDetail(BaseModel):
    """Detailed source information for a retrieved chunk."""
    source: str
    page_number: int | None = None
    heading: str | None = None

class GeneratedAnswer(BaseModel):
    """Result of the generation service."""
    text: str
    sources: List[str]  # List of unique source filenames (for backward compatibility)
    source_details: List[SourceDetail] = Field(default_factory=list) # New field for structured metadata
    model: str

@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM providers."""
    async def generate(self, prompt: str) -> str: ...

class GenerationService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def generate_answer(self, question: str, chunks: list[DocumentChunk]) -> GeneratedAnswer:
        prompt = self._construct_prompt(question, chunks)
        response = await self.llm_client.generate(prompt)

        unique_sources = sorted(list({c.source for c in chunks if c.source}))

        source_details = self._extract_source_details(chunks)

        return GeneratedAnswer(
            text=response,
            sources=unique_sources,
            source_details=source_details,
            model="qwen2.5:7b"
        )

    def _construct_prompt(self, question: str, chunks: list[DocumentChunk]) -> str:
        context_parts = []
        seen_chunk_ids = set()

        for chunk in chunks:
            if not chunk.source: continue
            if chunk.chunk_id in seen_chunk_ids: continue

            source_identifier = chunk.source
            page_info = ""
            if chunk.page_number is not None:
                page_info = f", Page: {chunk.page_number}"
            heading_info = f", Heading: {chunk.heading}" if chunk.heading else ""

            citation_tag = f"[Source: {source_identifier}{page_info}{heading_info}]"

            context_parts.append(f"{citation_tag}\n{chunk.text}")
            seen_chunk_ids.add(chunk.chunk_id)

        context = "\n\n".join(context_parts)

        return f"""You are a helpful assistant for Enterprise Knowledge. Your goal is to provide accurate, evidence-grounded answers to user questions.

**Instructions:**
1. Answer the user's question directly, using *only* the factual information provided in the "Context" section below.
2. Ground every factual claim in the retrieved context. Preserve material conditions, exceptions, thresholds, dates, eligibility requirements, and other qualifications from the source. Be especially careful with numerical values; only state them if directly supported.
3. For critical policy details, numerical values, or technical identifiers, prefer wording that is very close to the source text. For other information, a faithful paraphrase is acceptable if it improves clarity.
4. Cite every factual claim with its corresponding source using the format `[Source: document_name, Page: page_number_or_N/A]`. If a claim draws from multiple sources, cite all relevant ones. Never invent sources, page numbers, or identifiers.
5. If the "Context" does not contain enough information to answer the question, state clearly that the available documents do not provide enough information, rather than making assumptions or guessing.
6. Avoid adding generic claims, external knowledge, or conversational filler not supported by the provided documents.

**Context:**
{context}

**Question:**
{question}

**Answer:**
"""

    def _extract_source_details(self, chunks: list[DocumentChunk]) -> list[SourceDetail]:
        """Extracts structured source details, deduplicating by source and page/heading."""
        details_map = {}

        for chunk in chunks:
            if not chunk.source: continue

            key = (chunk.source, chunk.page_number, chunk.heading)
            if key not in details_map:
                details_map[key] = SourceDetail(
                    source=chunk.source,
                    page_number=chunk.page_number,
                    heading=chunk.heading
                )

        return sorted(details_map.values(), key=lambda x: (x.source, x.page_number or -1, x.heading or ""))
