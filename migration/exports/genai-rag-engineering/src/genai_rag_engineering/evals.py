"""Evaluation metrics for RAG predictions.

Three families, all returning an `EvalMetricResult` with a bootstrapped 95% CI:

- **Retrieval** (no LLM): Hit@k, MRR, nDCG@k, context recall vs gold_doc_ids.
- **Generation via LLM-as-judge**: faithfulness, answer_relevance, context_precision.
  Each judgement is parsed from strict JSON emitted by the judge LLM.
- **Judge calibration** (when ground-truth labels are available): Cohen's κ + Pearson r
  between judge score and oracle / human label, with bootstrap CIs.

All metrics accept a list of `Prediction` + matching list of `EvalRecord` and return an
`EvalMetricResult`. The `build_eval_report` helper wires them together into a single
`EvalReport` consumed by the service-delivery release gate.
"""

from __future__ import annotations

import json
import math
import re
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from .llm import LLMClient
from .prompts import REGISTRY as PROMPT_REGISTRY
from .schemas import (
    EvalMetricResult,
    EvalRecord,
    EvalReport,
    LLMMessage,
    Prediction,
)

# ----------------------------------------------------------------------------
# Bootstrap CI
# ----------------------------------------------------------------------------


def _bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via a percentile bootstrap over values."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(20260422)
    n = len(arr)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def _as_metric(name: str, per_item: list[float]) -> EvalMetricResult:
    mean, lo, hi = _bootstrap_ci(per_item)
    return EvalMetricResult(
        name=name,
        mean=mean,
        ci_low=lo,
        ci_high=hi,
        n=len(per_item),
        per_item=per_item,
    )


# ----------------------------------------------------------------------------
# Retrieval metrics (no LLM)
# ----------------------------------------------------------------------------


def _gold_hits(retrieved_doc_ids: list[str], gold_doc_ids: Iterable[str]) -> list[int]:
    """Binary relevance per retrieval rank."""
    gold = set(gold_doc_ids)
    return [1 if d in gold else 0 for d in retrieved_doc_ids]


def hit_at_k(preds: list[Prediction], records: list[EvalRecord], k: int) -> EvalMetricResult:
    per = []
    for p, r in zip(preds, records):
        if not r.gold_doc_ids:
            continue
        retrieved_doc_ids = [rc.doc_id for rc in p.retrieved[:k]]
        hits = _gold_hits(retrieved_doc_ids, r.gold_doc_ids)
        per.append(1.0 if any(hits) else 0.0)
    return _as_metric(f"retrieval.hit@{k}", per)


def mrr_at_k(preds: list[Prediction], records: list[EvalRecord], k: int = 10) -> EvalMetricResult:
    per = []
    for p, r in zip(preds, records):
        if not r.gold_doc_ids:
            continue
        retrieved_doc_ids = [rc.doc_id for rc in p.retrieved[:k]]
        hits = _gold_hits(retrieved_doc_ids, r.gold_doc_ids)
        rr = 0.0
        for rank, h in enumerate(hits, 1):
            if h:
                rr = 1.0 / rank
                break
        per.append(rr)
    return _as_metric(f"retrieval.mrr@{k}", per)


def ndcg_at_k(preds: list[Prediction], records: list[EvalRecord], k: int = 5) -> EvalMetricResult:
    per = []
    for p, r in zip(preds, records):
        if not r.gold_doc_ids:
            continue
        retrieved_doc_ids = [rc.doc_id for rc in p.retrieved[:k]]
        hits = _gold_hits(retrieved_doc_ids, r.gold_doc_ids)
        dcg = sum((2**h - 1) / math.log2(i + 2) for i, h in enumerate(hits))
        ideal_hits = sorted(hits, reverse=True)
        idcg = sum((2**h - 1) / math.log2(i + 2) for i, h in enumerate(ideal_hits))
        per.append(dcg / idcg if idcg > 0 else 0.0)
    return _as_metric(f"retrieval.ndcg@{k}", per)


def context_recall(preds: list[Prediction], records: list[EvalRecord], k: int = 10) -> EvalMetricResult:
    """Fraction of gold documents that appear in the top-k retrieved."""
    per = []
    for p, r in zip(preds, records):
        if not r.gold_doc_ids:
            continue
        retrieved = {rc.doc_id for rc in p.retrieved[:k]}
        gold = set(r.gold_doc_ids)
        per.append(len(retrieved & gold) / len(gold))
    return _as_metric(f"retrieval.context_recall@{k}", per)


def retrieval_metric_suite(
    preds: list[Prediction], records: list[EvalRecord], k_list: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[str, EvalMetricResult]:
    """Bundle hit/mrr/ndcg/context_recall at several ks."""
    out: dict[str, EvalMetricResult] = {}
    for k in k_list:
        out[f"hit@{k}"] = hit_at_k(preds, records, k)
        out[f"context_recall@{k}"] = context_recall(preds, records, k)
    out["mrr@10"] = mrr_at_k(preds, records, 10)
    out["ndcg@5"] = ndcg_at_k(preds, records, 5)
    return out


# ----------------------------------------------------------------------------
# Judge-based metrics
# ----------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_judge_json(text: str) -> dict:
    """Extract the first JSON object from the judge response; lenient on surrounding text."""
    m = _JSON_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _judge_call(judge: LLMClient, prompt_id: str, **kwargs) -> dict:
    prompt = PROMPT_REGISTRY.get(prompt_id)
    system, user = prompt.render(**kwargs)
    resp = judge.complete(
        [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
        max_tokens=160,
        temperature=0.0,
    )
    return _parse_judge_json(resp.text)


def faithfulness(preds: list[Prediction], judge: LLMClient) -> EvalMetricResult:
    """LLM-as-judge faithfulness: does every claim in the answer follow from the context?"""
    per: list[float] = []
    for p in preds:
        if p.answer.refused or not p.retrieved:
            per.append(1.0 if p.answer.refused else 0.0)
            continue
        context = "\n\n".join(f"[{r.doc_id}] {r.text}" for r in p.retrieved)
        v = _judge_call(
            judge,
            "judge.faithfulness",
            question=p.query,
            context=context,
            answer=p.answer.answer,
        )
        per.append(float(v.get("score", 0.0)))
    return _as_metric("generation.faithfulness", per)


def answer_relevance(preds: list[Prediction], judge: LLMClient) -> EvalMetricResult:
    per: list[float] = []
    for p in preds:
        v = _judge_call(
            judge,
            "judge.answer_relevance",
            question=p.query,
            answer=p.answer.answer,
        )
        per.append(float(v.get("score", 0.0)))
    return _as_metric("generation.answer_relevance", per)


def context_precision(preds: list[Prediction], judge: LLMClient, k: int = 5) -> EvalMetricResult:
    """Rank-weighted precision: how many of the top-k retrieved passages are relevant?"""
    per: list[float] = []
    for p in preds:
        if not p.retrieved:
            per.append(0.0)
            continue
        rels = []
        for r in p.retrieved[:k]:
            v = _judge_call(judge, "judge.context_precision", question=p.query, passage=r.text)
            rels.append(1.0 if v.get("relevant") in (True, "true", 1, "1") else 0.0)
        # Rank-weighted precision (standard RAGAS definition)
        if not rels or sum(rels) == 0:
            per.append(0.0)
            continue
        numerator = sum((sum(rels[: i + 1]) / (i + 1)) * rels[i] for i in range(len(rels)))
        per.append(numerator / sum(rels))
    return _as_metric(f"generation.context_precision@{k}", per)


# ----------------------------------------------------------------------------
# Judge-calibration helpers (used when we have oracle / human labels for a subset)
# ----------------------------------------------------------------------------


def judge_calibration(
    judge_scores: list[float],
    oracle_scores: list[float],
) -> dict[str, float | str]:
    """Report Cohen's κ (+quadratic), Pearson r, Spearman ρ between judge and oracle labels.

    If either input is constant (no variance) the correlation and κ are not defined;
    we return 0.0 for all four and add a `degenerate` note explaining why.
    """
    if len(judge_scores) < 3:
        return {
            "n": len(judge_scores),
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "cohen_kappa": 0.0,
            "cohen_kappa_quadratic": 0.0,
            "degenerate": "n < 3; insufficient sample for calibration",
        }

    judge_arr = np.asarray(judge_scores, dtype=float)
    oracle_arr = np.asarray(oracle_scores, dtype=float)
    if np.ptp(judge_arr) == 0.0 or np.ptp(oracle_arr) == 0.0:
        return {
            "n": len(judge_scores),
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "cohen_kappa": 0.0,
            "cohen_kappa_quadratic": 0.0,
            "degenerate": (
                "input is constant — correlation / kappa undefined. "
                "Extend the eval set with harder or noisier queries, "
                "or use a noisier label source, to populate calibration."
            ),
        }

    def _bucket(x: float) -> int:
        return 0 if x < 0.34 else (1 if x < 0.67 else 2)

    j_labels = [_bucket(x) for x in judge_scores]
    o_labels = [_bucket(x) for x in oracle_scores]
    kappa = float(cohen_kappa_score(o_labels, j_labels))
    try:
        kappa_q = float(cohen_kappa_score(o_labels, j_labels, weights="quadratic"))
    except Exception:
        kappa_q = 0.0
    r, _ = pearsonr(judge_scores, oracle_scores)
    rho, _ = spearmanr(judge_scores, oracle_scores)
    return {
        "n": len(judge_scores),
        "pearson_r": float(r) if np.isfinite(r) else 0.0,
        "spearman_rho": float(rho) if np.isfinite(rho) else 0.0,
        "cohen_kappa": 0.0 if not np.isfinite(kappa) else kappa,
        "cohen_kappa_quadratic": 0.0 if not np.isfinite(kappa_q) else kappa_q,
    }


# ----------------------------------------------------------------------------
# Report builder + release-gate logic
# ----------------------------------------------------------------------------

DEFAULT_GATE = {
    "min_hit_at_5": 0.70,
    "min_mrr_at_10": 0.55,
    "min_faithfulness": 0.70,
    "min_answer_relevance": 0.70,
    "max_p95_latency_ms": 5000.0,
}


def build_eval_report(
    preds_by_retriever: dict[str, list[Prediction]],
    records: list[EvalRecord],
    judge: LLMClient | None = None,
    *,
    gate: dict[str, float] | None = None,
    pipeline_commit: str = "",
) -> EvalReport:
    """Aggregate metrics across retrievers and produce a release-gate decision."""
    gate = gate or DEFAULT_GATE
    retrieval: dict[str, dict[str, EvalMetricResult]] = {}
    generation: dict[str, EvalMetricResult] = {}
    latency_by_retriever: dict[str, list[float]] = {}
    cost_by_retriever: dict[str, list[float]] = {}

    primary = max(preds_by_retriever.keys(), key=lambda r: len(preds_by_retriever[r]))
    for retriever_name, preds in preds_by_retriever.items():
        retrieval[retriever_name] = retrieval_metric_suite(preds, records)
        latency_by_retriever[retriever_name] = [p.latency_ms_total for p in preds]
        cost_by_retriever[retriever_name] = [p.cost_usd for p in preds]

    # Generation metrics: one set, computed against the primary retriever's predictions.
    if judge is not None:
        primary_preds = preds_by_retriever[primary]
        generation["faithfulness"] = faithfulness(primary_preds, judge)
        generation["answer_relevance"] = answer_relevance(primary_preds, judge)
        generation["context_precision"] = context_precision(primary_preds, judge)

    # Summaries
    n_queries = max(len(p) for p in preds_by_retriever.values())
    latency_summary = {f"{r}_p50_ms": float(np.median(v)) if v else 0.0 for r, v in latency_by_retriever.items()}
    latency_summary.update(
        {f"{r}_p95_ms": float(np.quantile(v, 0.95)) if v else 0.0 for r, v in latency_by_retriever.items()}
    )
    cost_summary = {f"{r}_total_usd": float(sum(v)) for r, v in cost_by_retriever.items()}
    cost_summary["grand_total_usd"] = sum(cost_summary.values())

    # Gate decision — against the primary retriever
    rationale: list[str] = []
    ok = True
    primary_metrics = retrieval[primary]
    checks: list[tuple[str, float, float, str]] = [
        ("hit@5", primary_metrics["hit@5"].mean, gate["min_hit_at_5"], ">="),
        ("mrr@10", primary_metrics["mrr@10"].mean, gate["min_mrr_at_10"], ">="),
    ]
    if "faithfulness" in generation:
        checks.append(("faithfulness", generation["faithfulness"].mean, gate["min_faithfulness"], ">="))
    if "answer_relevance" in generation:
        checks.append(("answer_relevance", generation["answer_relevance"].mean, gate["min_answer_relevance"], ">="))
    checks.append(("p95_latency_ms", latency_summary.get(f"{primary}_p95_ms", 0.0), gate["max_p95_latency_ms"], "<="))

    for name, observed, threshold, op in checks:
        if op == ">=":
            passed = observed >= threshold
        else:
            passed = observed <= threshold
        if not passed:
            ok = False
        rationale.append(f"{'PASS' if passed else 'FAIL'}: {name}={observed:.3f} {op} {threshold:.3f}")

    return EvalReport(
        n_queries=n_queries,
        retrievers_compared=list(preds_by_retriever.keys()),
        retrieval_metrics=retrieval,
        generation_metrics=generation,
        cost_summary=cost_summary,
        latency_summary=latency_summary,
        pipeline_commit=pipeline_commit,
        passes_gate=ok,
        gate_rationale=rationale,
    )
