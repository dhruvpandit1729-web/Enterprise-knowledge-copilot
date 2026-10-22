import pytest
from app.evals.metrics import (
    EvalExample,
    compute_retrieval_recall_diagnostics,
    compute_citation_validity_diagnostics,
    compute_answerability_accuracy,
    compute_key_fact_coverage_diagnostics,
    compute_claim_support_score,
    build_example_diagnostics,
    check_abstention,
)
from app.schemas.document import DocumentChunk


def test_retrieval_recall_diagnostics():
    chunk1 = DocumentChunk(
        chunk_id="c1", doc_id="d1", source="doc1.md", text="text1",
        token_count=5, section_index=0, chunk_index=0
    )
    chunk2 = DocumentChunk(
        chunk_id="c2", doc_id="d2", source="doc2.md", text="text2",
        token_count=5, section_index=0, chunk_index=1
    )

    recall, ret, matched, missing = compute_retrieval_recall_diagnostics(
        expected_sources=["doc1.md", "doc3.md"],
        retrieved_chunks=[chunk1, chunk2],
    )
    assert recall == 0.5
    assert "doc1.md" in matched
    assert "doc3.md" in missing
    assert set(ret) == {"doc1.md", "doc2.md"}


def test_citation_validity_diagnostics():
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", source="policy.md", text="text",
        token_count=5, section_index=0, chunk_index=0
    )

    score, valid, invalid = compute_citation_validity_diagnostics(
        answer_sources=["policy.md", "fake.md"],
        retrieved_chunks=[chunk],
    )
    assert score == 0.5
    assert valid == ["policy.md"]
    assert invalid == ["fake.md"]


def test_key_fact_coverage_diagnostics_token_fallback():
    # Tests matching even if punctuation/formatting differs (e.g., "$500" vs "500 per month")
    score, matched, missed = compute_key_fact_coverage_diagnostics(
        expected_facts=["500 per month", "30 days"],
        answer_text="Reimbursement is up to $500, per month within 30 days.",
    )
    assert score == 1.0
    assert len(matched) == 2
    assert len(missed) == 0


def test_check_abstention_and_answerability():
    abstaining_text = "The available documents do not provide enough information on this topic."
    answering_text = "Employees are allowed up to two days remote per week."

    assert check_abstention(abstaining_text) is True
    assert check_abstention(answering_text) is False

    assert compute_answerability_accuracy(True, answering_text) == 1.0
    assert compute_answerability_accuracy(True, abstaining_text) == 0.0
    assert compute_answerability_accuracy(False, abstaining_text) == 1.0
    assert compute_answerability_accuracy(False, answering_text) == 0.0


def test_claim_support_score_faithful_paraphrase():
    # Inspired by heldout_002: accurate answer with some paraphrasing
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", source="policy.md",
        text="Standard business travel and hardware expenses incurred for official company business are fully reimbursable up to $500 per month, provided original receipts are submitted within 30 days of purchase.",
        token_count=18, section_index=0, chunk_index=0
    )
    answer_text = "Yes, a hardware purchase of $400 can be reimbursed. The policy allows up to $500 monthly for such expenses, given original receipts are submitted within 30 days. [Source: policy.md]"

    # With adjusted thresholds, this should now pass closer to 1.0 if truly supported
    score = compute_claim_support_score(answer_text, [chunk], min_token_overlap_ratio=0.3, min_sentence_len=2, min_claim_token_len=1)
    assert score >= 0.75 # Expecting higher due to prompt changes and adjusted metric tolerance

def test_claim_support_score_unsupported_claim():
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", source="policy.md",
        text="Remote work is allowed up to two days a week.",
        token_count=10, section_index=0, chunk_index=0
    )
    answer_text = "Employees get unlimited free coffee and snacks. Remote work is two days a week. [Source: policy.md]"
    score = compute_claim_support_score(answer_text, [chunk])
    # First sentence is unsupported, second is supported. Expect score around 0.5.
    assert score == 0.5


def test_build_example_diagnostics():
    ex = EvalExample(
        id="ex_test",
        question="What is the policy?",
        expected_answerable=True,
        expected_sources=["policy.md"],
        expected_facts=["two days"],
    )
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", source="policy.md", text="two days remote",
        token_count=3, section_index=0, chunk_index=0
    )
    diag = build_example_diagnostics(
        example=ex,
        retrieved_chunks=[chunk],
        answer_text="It allows two days. [Source: policy.md]",
        answer_sources=["policy.md"],
    )

    assert diag.example_id == "ex_test"
    assert diag.retrieval_recall == 1.0
    assert diag.citation_validity == 1.0
    assert diag.key_fact_coverage == 1.0
    assert diag.claim_support_score == 1.0
    assert diag.claim_support_applicable is True
    assert diag.answerability_correct is True
    assert len(diag.failure_reasons) == 0


def test_unanswerable_correct_abstention_claim_support_na():
    ex = EvalExample(
        id="ex_unans",
        question="What is the secret code?",
        expected_answerable=False,
        expected_sources=[],
        expected_facts=[],
    )
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", source="policy.md", text="remote work rules",
        token_count=3, section_index=0, chunk_index=0
    )
    diag = build_example_diagnostics(
        example=ex,
        retrieved_chunks=[chunk],
        answer_text="The available documents do not provide enough information. [Source: policy.md]",
        answer_sources=["policy.md"],
    )

    assert diag.answerability_correct is True
    assert diag.claim_support_applicable is False
    assert diag.claim_support_score is None


def test_unanswerable_incorrect_attempt_claim_support_scored():
    ex = EvalExample(
        id="ex_unans_bad",
        question="What is the secret code?",
        expected_answerable=False,
        expected_sources=[],
        expected_facts=[],
    )
    chunk = DocumentChunk(
        chunk_id="c1", doc_id="d1", source="policy.md", text="remote work rules",
        token_count=3, section_index=0, chunk_index=0
    )
    diag = build_example_diagnostics(
        example=ex,
        retrieved_chunks=[chunk],
        answer_text="The secret code is 12345. [Source: policy.md]",
        answer_sources=["policy.md"],
    )

    assert diag.answerability_correct is False
    assert diag.claim_support_applicable is True
    assert diag.claim_support_score == 0.0
