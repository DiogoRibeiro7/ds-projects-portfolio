from __future__ import annotations

from genai_rag_engineering.evals import hit_at_k, mrr_at_k, retrieval_metric_suite
from genai_rag_engineering.retrieval import reciprocal_rank_fusion
from genai_rag_engineering.schemas import (
    EvalRecord,
    GeneratedAnswer,
    GuardrailResult,
    LLMResponse,
    Prediction,
    RetrievedChunk,
)


def _prediction(retrieved: list[RetrievedChunk]) -> Prediction:
    return Prediction(
        request_id="req-1",
        tenant_id="default",
        query="refund policy",
        prompt_id="rag.answer",
        prompt_version="v1",
        retriever="dense",
        retrieved=retrieved,
        answer=GeneratedAnswer(answer="Refunds are available. Citations: [policy-a]."),
        llm=LLMResponse(text="Refunds are available.", model="fake-default", provider="fake"),
        guardrails_input=GuardrailResult(),
        guardrails_output=GuardrailResult(),
    )


def test_reciprocal_rank_fusion_merges_ranked_results() -> None:
    dense = [
        RetrievedChunk(
            chunk_id="a",
            doc_id="policy-a",
            text="refund policy",
            score=1.0,
            retriever="dense",
            rank=0,
        )
    ]
    sparse = [
        RetrievedChunk(
            chunk_id="b",
            doc_id="policy-b",
            text="shipping policy",
            score=1.0,
            retriever="bm25",
            rank=0,
        ),
        RetrievedChunk(
            chunk_id="a",
            doc_id="policy-a",
            text="refund policy",
            score=0.5,
            retriever="bm25",
            rank=1,
        ),
    ]

    fused = reciprocal_rank_fusion([dense, sparse], k=2)

    assert [item.chunk_id for item in fused] == ["a", "b"]
    assert fused[0].retriever == "hybrid"


def test_retrieval_metrics_score_gold_hits() -> None:
    preds = [
        _prediction(
            [
                RetrievedChunk(
                    chunk_id="a",
                    doc_id="policy-a",
                    text="refund policy",
                    score=0.9,
                    retriever="dense",
                    rank=0,
                )
            ]
        )
    ]
    records = [
        EvalRecord(
            query_id="q1",
            query="refund policy",
            gold_answer="Refunds are available.",
            gold_doc_ids=["policy-a"],
        )
    ]

    assert hit_at_k(preds, records, k=1).mean == 1.0
    assert mrr_at_k(preds, records, k=1).mean == 1.0
    suite = retrieval_metric_suite(preds, records, k_list=(1,))
    assert set(suite) == {"hit@1", "mrr@10", "ndcg@5", "context_recall@1"}
