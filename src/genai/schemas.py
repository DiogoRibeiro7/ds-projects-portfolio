"""Pydantic schemas for the RAG pipeline, evaluation, and service-delivery layers.

Every record that crosses a notebook boundary is a `Prediction` or `EvalReport` — both
serialisable to JSONL / JSON — so that the three notebooks can run independently but
compose into a single pipeline (pipeline → eval → service).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------- Knowledge base + retrieval ----------

class Document(BaseModel):
    """A raw document ingested into the knowledge base."""
    model_config = ConfigDict(frozen=True)
    doc_id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk of a document produced by the chunker."""
    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Query(BaseModel):
    """Inbound query to the RAG service."""
    text: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = "default"
    request_id: str = Field(default_factory=_uuid)
    k: int = Field(default=5, ge=1, le=25)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its score, source retriever, and rank."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    retriever: Literal["bm25", "dense", "hybrid", "reranked"]
    rank: int


class Citation(BaseModel):
    """A span-level citation emitted by the generator."""
    chunk_id: str
    doc_id: str
    quote: str = ""


# ---------- LLM ----------

class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: Literal["anthropic", "openai", "fake"]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str = ""


class GeneratedAnswer(BaseModel):
    """Structured output from the generator."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""


# ---------- Guardrails ----------

class GuardrailFinding(BaseModel):
    name: str
    severity: Literal["info", "warn", "block"]
    matched: str = ""
    reason: str = ""


class GuardrailResult(BaseModel):
    blocked: bool = False
    findings: list[GuardrailFinding] = Field(default_factory=list)
    redacted_text: str = ""


# ---------- Pipeline I/O ----------

class Prediction(BaseModel):
    """A complete pipeline prediction.

    Written to JSONL by `genai_rag_pipeline.ipynb`; read (+ scored) by
    `llm_rag_evaluation.ipynb`; archived for audit by `genai_service_delivery.ipynb`.
    """
    request_id: str
    tenant_id: str
    query: str
    prompt_id: str
    prompt_version: str
    retriever: str                                     # label of the retrieval strategy
    retrieved: list[RetrievedChunk]
    answer: GeneratedAnswer
    llm: LLMResponse
    guardrails_input: GuardrailResult = Field(default_factory=GuardrailResult)
    guardrails_output: GuardrailResult = Field(default_factory=GuardrailResult)
    cost_usd: float = 0.0
    latency_ms_total: float = 0.0
    timestamp: str = Field(default_factory=_now_iso)
    trace: list[dict[str, Any]] = Field(default_factory=list)


# ---------- Evaluation ----------

class EvalRecord(BaseModel):
    """One labeled example in the evaluation dataset."""
    query_id: str
    query: str
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_answer: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: list[str] = Field(default_factory=list)


class EvalMetricResult(BaseModel):
    """One metric's aggregate result with a bootstrapped 95% CI."""
    name: str
    mean: float
    ci_low: float
    ci_high: float
    n: int
    per_item: list[float] = Field(default_factory=list, repr=False)


class EvalReport(BaseModel):
    """Comprehensive eval output consumed by the service-delivery release gate."""
    n_queries: int
    retrievers_compared: list[str] = Field(default_factory=list)
    retrieval_metrics: dict[str, dict[str, EvalMetricResult]] = Field(default_factory=dict)
    generation_metrics: dict[str, EvalMetricResult] = Field(default_factory=dict)
    judge_calibration: dict[str, Any] = Field(default_factory=dict)
    cost_summary: dict[str, float] = Field(default_factory=dict)
    latency_summary: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)
    pipeline_commit: str = ""
    passes_gate: bool = False
    gate_rationale: list[str] = Field(default_factory=list)
