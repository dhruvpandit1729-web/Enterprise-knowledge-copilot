from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.evals.metrics import (
    AggregateEvalReport,
    EvalDataset,
    build_example_diagnostics,
)
from app.generation.client import OllamaClient
from app.generation.service import GenerationService
from app.retrieval.store import get_default_store
from app.service import QuestionAnsweringService


async def run_evaluation(dataset_path: str, limit: int = 5, output_report: str | None = None) -> AggregateEvalReport:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = EvalDataset(**data)

    store = get_default_store()
    qa_service = QuestionAnsweringService(
        document_store=store,
        generation_service=GenerationService(llm_client=OllamaClient()),
    )

    diagnostics_list = []
    print(f"Starting evaluation of {len(dataset.examples)} examples (dataset v{dataset.dataset_version})...\n")

    for ex in dataset.examples:
        # Retrieve chunks
        retrieved_chunks = store.search(ex.question, limit=limit)

        # Generate answer
        answer = await qa_service.answer_question(ex.question, limit=limit)

        # Build detailed diagnostics
        diag = build_example_diagnostics(
            example=ex,
            retrieved_chunks=retrieved_chunks,
            answer_text=answer.text,
            answer_sources=answer.sources,
        )
        diagnostics_list.append(diag)

        # Print per-example summary
        print(f"[{diag.example_id}] Question: {diag.question}")
        print(f"  - Expected Answerable: {diag.expected_answerable} | Correct Answerability: {diag.answerability_correct}")
        print(f"  - Retrieval Recall: {diag.retrieval_recall:.2f} (Retrieved: {diag.retrieved_sources})")
        print(f"  - Citation Validity: {diag.citation_validity:.2f} (Citations: {diag.answer_sources})")
        print(f"  - Key-Fact Coverage: {diag.key_fact_coverage:.2f} (Matched: {diag.matched_facts}, Missed: {diag.missed_facts})")
        claim_str = f"{diag.claim_support_score:.2f}" if diag.claim_support_score is not None else "N/A"
        print(f"  - Claim Support Score: {claim_str}")
        if diag.failure_reasons:
            print(f"  - Failures: {diag.failure_reasons}")
        print()

    total = len(diagnostics_list)
    mean_recall = sum(d.retrieval_recall for d in diagnostics_list) / total
    mean_cite = sum(d.citation_validity for d in diagnostics_list) / total
    mean_ans = sum(1.0 if d.answerability_correct else 0.0 for d in diagnostics_list) / total
    mean_facts = sum(d.key_fact_coverage for d in diagnostics_list) / total

    # Compute claim support only over applicable examples
    applicable_claim_scores = [
        d.claim_support_score for d in diagnostics_list if d.claim_support_applicable and d.claim_support_score is not None
    ]
    applicable_count = len(applicable_claim_scores)
    na_count = total - applicable_count
    mean_claim = (sum(applicable_claim_scores) / applicable_count) if applicable_count > 0 else None

    report = AggregateEvalReport(
        dataset_version=dataset.dataset_version,
        total_examples=total,
        mean_retrieval_recall=mean_recall,
        mean_citation_validity=mean_cite,
        mean_answerability_accuracy=mean_ans,
        mean_key_fact_coverage=mean_facts,
        mean_claim_support=mean_claim,
        claim_support_applicable_count=applicable_count,
        claim_support_na_count=na_count,
        diagnostics=diagnostics_list,
    )

    print("==================== Evaluation Summary ====================")
    print(f"Dataset Version: {dataset.dataset_version}")
    print(f"Total Examples Evaluated: {total}")
    print(f"Mean Retrieval Recall@{limit}: {mean_recall:.2%}")
    print(f"Mean Citation Validity: {mean_cite:.2%}")
    print(f"Mean Answerability Accuracy: {mean_ans:.2%}")
    print(f"Mean Key-Fact Coverage: {mean_facts:.2%}")
    claim_disp = f"{mean_claim:.2%}" if mean_claim is not None else "N/A"
    print(f"Mean Claim Support Score: {claim_disp} (Applicable: {applicable_count}, Excluded/N/A: {na_count})")
    print("============================================================")

    if output_report:
        out_path = Path(output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"\nDetailed report saved to: {out_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/datasets/rag_eval_dataset.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default="evals/reports/latest_eval_report.json")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.dataset, limit=args.limit, output_report=args.output))
