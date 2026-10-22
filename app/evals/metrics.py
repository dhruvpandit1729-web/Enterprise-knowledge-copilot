from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field


class EvalExample(BaseModel):
    id: str
    question: str
    expected_answerable: bool
    expected_sources: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)


class EvalDataset(BaseModel):
    dataset_version: str
    description: str
    examples: list[EvalExample]


class ExampleDiagnostics(BaseModel):
    example_id: str
    question: str
    expected_answerable: bool
    retrieved_sources: list[str]
    matched_sources: list[str]
    missing_sources: list[str]
    expected_sources: list[str]
    answer_sources: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    expected_facts: list[str]
    matched_facts: list[str]
    missed_facts: list[str]
    answer_text: str
    is_abstaining: bool
    answerability_correct: bool
    retrieval_recall: float
    citation_validity: float
    key_fact_coverage: float
    claim_support_score: float | None = None
    claim_support_applicable: bool = True
    failure_reasons: list[str] = Field(default_factory=list)


class AggregateEvalReport(BaseModel):
    dataset_version: str
    total_examples: int
    mean_retrieval_recall: float
    mean_citation_validity: float
    mean_answerability_accuracy: float
    mean_key_fact_coverage: float
    mean_claim_support: float | None = None
    claim_support_applicable_count: int
    claim_support_na_count: int
    diagnostics: list[ExampleDiagnostics]


def compute_retrieval_recall_diagnostics(
    expected_sources: list[str],
    retrieved_chunks: list[Any],
) -> tuple[float, list[str], list[str], list[str]]:
    """Computes Retrieval Recall@k and returns (recall, retrieved_sources, matched, missing)."""
    retrieved_sources = sorted({getattr(c, "source", "") for c in retrieved_chunks if getattr(c, "source", "")})
    if not expected_sources:
        return 1.0, retrieved_sources, [], []

    matched = [s for s in expected_sources if s in retrieved_sources]
    missing = [s for s in expected_sources if s not in retrieved_sources]
    recall = len(matched) / len(expected_sources)
    return recall, retrieved_sources, matched, missing


def compute_retrieval_recall(expected_sources: list[str], retrieved_chunks: list[Any]) -> float:
    """Computes Retrieval Recall@k: fraction of expected sources found in retrieved chunks."""
    recall, _, _, _ = compute_retrieval_recall_diagnostics(expected_sources, retrieved_chunks)
    return recall


def compute_citation_validity_diagnostics(
    answer_sources: list[str],
    retrieved_chunks: list[Any],
) -> tuple[float, list[str], list[str]]:
    """Computes citation validity and returns (validity, valid_citations, invalid_citations)."""
    valid_retrieved_sources = {getattr(c, "source", "") for c in retrieved_chunks if getattr(c, "source", "")}
    if not answer_sources:
        return 1.0, [], []

    valid = [s for s in answer_sources if s in valid_retrieved_sources]
    invalid = [s for s in answer_sources if s not in valid_retrieved_sources]
    score = len(valid) / len(answer_sources)
    return score, valid, invalid


def compute_citation_validity(answer_sources: list[str], retrieved_chunks: list[Any]) -> float:
    """Computes citation validity: fraction of answer sources present in retrieved chunks."""
    score, _, _ = compute_citation_validity_diagnostics(answer_sources, retrieved_chunks)
    return score


def check_abstention(answer_text: str) -> bool:
    """Determines whether an answer expresses insufficient context or abstention."""
    text_lower = answer_text.lower()
    return any(
        phrase in text_lower
        for phrase in [
            "not provide enough information",
            "not provide sufficient information",
            "does not contain",
            "do not contain",
            "do not provide",
            "no information",
            "insufficient information",
        ]
    )


def compute_answerability_accuracy(expected_answerable: bool, answer_text: str) -> float:
    """Computes whether answerability behavior matches expectations."""
    is_abstaining = check_abstention(answer_text)
    if expected_answerable:
        return 0.0 if is_abstaining else 1.0
    return 1.0 if is_abstaining else 0.0


def _normalize_tokens(text: str) -> set[str]:
    """Extracts alphanumeric words/tokens in lowercase for flexible matching."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def compute_key_fact_coverage_diagnostics(
    expected_facts: list[str],
    answer_text: str,
) -> tuple[float, list[str], list[str]]:
    """Computes key-fact coverage and returns (score, matched_facts, missed_facts).

    Uses substring matching first; falls back to token subset overlap to handle
    formatting variants (such as '$500 per month' vs '500, per month').
    """
    if not expected_facts:
        return 1.0, [], []

    text_lower = answer_text.lower()
    answer_tokens = _normalize_tokens(answer_text)

    matched: list[str] = []
    missed: list[str] = []

    for fact in expected_facts:
        fact_lower = fact.lower().strip()
        # Direct substring check
        if fact_lower in text_lower:
            matched.append(fact)
            continue

        # Token overlap check (all tokens in expected fact present in answer)
        fact_tokens = _normalize_tokens(fact)
        if fact_tokens and fact_tokens.issubset(answer_tokens):
            matched.append(fact)
        else:
            missed.append(fact)

    score = len(matched) / len(expected_facts)
    return score, matched, missed


def compute_key_fact_coverage(expected_facts: list[str], answer_text: str) -> float:
    """Computes key-fact coverage: fraction of expected key facts found in answer text.

    This metric primarily measures lexical overlap for specific factual phrases.
    """
    score, _, _ = compute_key_fact_coverage_diagnostics(expected_facts, answer_text)
    return score


# NOTE: compute_answer_faithfulness is now explicitly deprecated in favor of
# compute_key_fact_coverage (lexical) and potentially a future semantic claim support metric.
# This alias is retained for backward compatibility with older reports, but new code
# should use compute_key_fact_coverage.
compute_answer_faithfulness = compute_key_fact_coverage


def compute_claim_support_score(
    answer_text: str,
    retrieved_chunks: list[Any],
    min_token_overlap_ratio: float = 0.4,
    min_sentence_len: int = 3,
    min_claim_token_len: int = 2,
) -> float:
    """Deterministic claim support metric: verifies sentence tokens against retrieved chunk context.

    Measures the proportion of content-bearing sentences in the answer whose tokens
    are supported by the supplied chunks, excluding explicit citation markers.
    This is a lexical overlap check and does not perform deep semantic entailment.
    """
    if not retrieved_chunks:
        return 0.0

    combined_context = " ".join([getattr(c, "text", "") for c in retrieved_chunks]).lower()
    context_tokens = _normalize_tokens(combined_context)
    if not context_tokens:
        return 0.0

    # Strip citation tags like [Source: ..., Page: ...]
    cleaned_answer = re.sub(r"\[Source:[^\]]*\]", "", answer_text, flags=re.IGNORECASE)
    sentences = [
        s.strip() for s in re.split(r"[.!?\n]+", cleaned_answer) if len(s.strip()) > min_sentence_len
    ]

    if not sentences:
        return 1.0

    supported_sentences = 0
    for s in sentences:
        s_tokens = [w for w in _normalize_tokens(s) if len(w) > min_claim_token_len]
        if not s_tokens:
            supported_sentences += 1  # Trivial sentences (e.g., just stop words) count as supported
            continue
        match_count = sum(1 for t in s_tokens if t in context_tokens)
        if (match_count / len(s_tokens)) >= min_token_overlap_ratio:
            supported_sentences += 1

    return supported_sentences / len(sentences)


def build_example_diagnostics(
    example: EvalExample,
    retrieved_chunks: list[Any],
    answer_text: str,
    answer_sources: list[str],
) -> ExampleDiagnostics:
    """Produces comprehensive diagnostics for an evaluation example."""
    recall, ret_sources, matched_sources, missing_sources = compute_retrieval_recall_diagnostics(
        example.expected_sources, retrieved_chunks
    )
    cite_validity, valid_cites, invalid_cites = compute_citation_validity_diagnostics(
        answer_sources, retrieved_chunks
    )
    is_abstaining = check_abstention(answer_text)
    answerability_correct = bool(compute_answerability_accuracy(example.expected_answerable, answer_text) == 1.0)
    fact_coverage, matched_facts, missed_facts = compute_key_fact_coverage_diagnostics(
        example.expected_facts, answer_text
    )

    # Determine claim support applicability:
    # If the question is unanswerable and the model correctly abstains, claim support is N/A (not applicable).
    # If the question is unanswerable and the model incorrectly attempts to answer, claim support is applicable and scores 0.0 (unsupported).
    # If the question is answerable, claim support is always applicable.
    claim_support_applicable = True
    if not example.expected_answerable and is_abstaining:
        claim_support_applicable = False
        claim_support = None
    else:
        claim_support = compute_claim_support_score(answer_text, retrieved_chunks)

    failures: list[str] = []
    if missing_sources:
        failures.append(f"Missing expected sources in retrieval: {missing_sources}")
    if invalid_cites:
        failures.append(f"Answer cited unretrieved sources: {invalid_cites}")
    if not answerability_correct:
        if example.expected_answerable:
            failures.append("Expected answerable question, but model abstained.")
        else:
            failures.append("Expected unanswerable question, but model attempted to answer.")
    if missed_facts:
        failures.append(f"Missing key facts: {missed_facts}")
    if claim_support_applicable and claim_support is not None and claim_support < 0.5:
        failures.append(f"Low claim support score: {claim_support:.2f}")

    return ExampleDiagnostics(
        example_id=example.id,
        question=example.question,
        expected_answerable=example.expected_answerable,
        retrieved_sources=ret_sources,
        matched_sources=matched_sources,
        missing_sources=missing_sources,
        expected_sources=example.expected_sources,
        answer_sources=answer_sources,
        valid_citations=valid_cites,
        invalid_citations=invalid_cites,
        expected_facts=example.expected_facts,
        matched_facts=matched_facts,
        missed_facts=missed_facts,
        answer_text=answer_text,
        is_abstaining=is_abstaining,
        answerability_correct=answerability_correct,
        retrieval_recall=recall,
        citation_validity=cite_validity,
        key_fact_coverage=fact_coverage,
        claim_support_score=claim_support,
        claim_support_applicable=claim_support_applicable,
        failure_reasons=failures,
    )
