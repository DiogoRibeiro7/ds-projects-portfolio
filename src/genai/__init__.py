"""Production-style GenAI primitives: schemas, LLM clients, embedders, chunking,
retrieval, guardrails, telemetry, prompts, pipeline orchestrator, and eval metrics.
Shared across the three GenAI notebooks (pipeline, evaluation, service-delivery).
"""
from __future__ import annotations

from .schemas import (
    Document, Chunk, Query, RetrievedChunk, Citation,
    LLMMessage, LLMResponse, GeneratedAnswer,
    GuardrailFinding, GuardrailResult,
    Prediction, EvalRecord, EvalMetricResult, EvalReport,
)
from .chunking import chunk_document, chunk_documents, chunk_size_stats
from .embeddings import SentenceTransformerEmbedder, get_embedder
from .guardrails import InputGuardrail, OutputGuardrail, REFUSAL_MESSAGE, detect_pii, redact_pii
from .llm import AnthropicClient, FakeLLMClient, OpenAIClient, get_llm_client
from .pipeline import RAGPipeline
from .prompts import Prompt, PromptRegistry, REGISTRY as PROMPT_REGISTRY
from .retrieval import (
    BM25Retriever, CrossEncoderReranker, DenseRetriever, HybridRetriever,
    hyde_transform, multi_query_rewrite, reciprocal_rank_fusion,
)
from .telemetry import CostTracker, RollingLatencyTracker, Tracer, structured_log

__all__ = [
    # schemas
    "Document", "Chunk", "Query", "RetrievedChunk", "Citation",
    "LLMMessage", "LLMResponse", "GeneratedAnswer",
    "GuardrailFinding", "GuardrailResult",
    "Prediction", "EvalRecord", "EvalMetricResult", "EvalReport",
    # chunking
    "chunk_document", "chunk_documents", "chunk_size_stats",
    # embeddings
    "SentenceTransformerEmbedder", "get_embedder",
    # guardrails
    "InputGuardrail", "OutputGuardrail", "REFUSAL_MESSAGE", "detect_pii", "redact_pii",
    # llm
    "AnthropicClient", "FakeLLMClient", "OpenAIClient", "get_llm_client",
    # pipeline
    "RAGPipeline",
    # prompts
    "Prompt", "PromptRegistry", "PROMPT_REGISTRY",
    # retrieval
    "BM25Retriever", "CrossEncoderReranker", "DenseRetriever", "HybridRetriever",
    "hyde_transform", "multi_query_rewrite", "reciprocal_rank_fusion",
    # telemetry
    "CostTracker", "RollingLatencyTracker", "Tracer", "structured_log",
]
