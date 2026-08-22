# GenAI RAG Engineering Report

## Scope

This report covers the copy-first export at:

`migration/exports/genai-rag-engineering`

The export packages the portfolio GenAI/RAG code as a standalone-ready repository
candidate. Original `src/genai`, GenAI notebooks, and generated artifacts remain in the
portfolio repository.

## Architecture

The export tells a coherent RAG engineering story:

`schemas -> chunking -> embeddings -> retrieval/reranking -> prompts -> generation -> guardrails -> evaluation -> telemetry -> pipeline`

Main package namespace:

`genai_rag_engineering`

The export rewrites the monorepo-facing `src.genai` namespace into this package namespace.
CI and tests use deterministic offline components, so provider credentials are not required.

## Migrated Components

| Module | Role |
|---|---|
| `schemas.py` | Pydantic models for documents, chunks, retrieved chunks, citations, LLM messages/responses, predictions, guardrail results, and evaluation reports. |
| `chunking.py` | Recursive document splitting, multi-document chunking, and chunk-size diagnostics. |
| `embeddings.py` | Embedder protocol and optional sentence-transformer embedder factory. |
| `retrieval.py` | dense retriever, BM25 retriever, hybrid retrieval, reciprocal-rank fusion, cross-encoder reranking, HyDE, and multi-query rewrite helpers. |
| `prompts.py` | prompt dataclass, prompt registry, and default prompt templates. |
| `llm.py` | LLM client protocol, deterministic `FakeLLMClient`, optional OpenAI client, and optional Anthropic client. |
| `guardrails.py` | PII detection/redaction, prompt-injection checks, input guardrail, and output guardrail. |
| `evals.py` | retrieval metrics, judge-based metrics, calibration helpers, and report assembly. |
| `telemetry.py` | tracing spans, cost tracking, latency tracking, and structured logging. |
| `pipeline.py` | `RAGPipeline` orchestration over retrieval, optional reranking, prompt rendering, LLM call, guardrails, citations, and prediction schema. |

## Notebook Examples

Copied notebooks:

- `notebooks/genai_rag_pipeline.ipynb`
- `notebooks/llm_rag_evaluation.ipynb`
- `notebooks/genai_service_delivery.ipynb`
- `notebooks/genai_dataops_vector_platform.ipynb`

These notebooks were retained because they form a coherent progression from RAG pipeline
construction through evaluation, service delivery, and vector/data operations. Generated
vector indexes, predictions, audit logs, and deployment artifacts under `artifacts/genai`
were not copied.

## Components Left Behind

The export intentionally leaves behind:

- unrelated generic API, cloud, security, privacy, compliance, AutoML, and deployment frameworks;
- generated FAISS/vector-store artifacts, predictions, audit logs, and runtime files;
- root monorepo package/dependency configuration;
- portfolio docs and tests not directly tied to the GenAI/RAG package;
- live-provider credentials or environment-specific configuration.

## Dependency Set

Runtime dependencies:

- `numpy`
- `pydantic`
- `scikit-learn`
- `scipy`

Development dependencies:

- `mypy`
- `pytest`
- `pytest-cov`
- `ruff`

Optional extras:

- `live`: `openai`, `anthropic`
- `retrieval`: `faiss-cpu`, `rank-bm25`, `sentence-transformers`
- `service`: `fastapi`, `uvicorn`

The default dependency set is intentionally offline-testable and avoids model downloads.

## Public API

The package-level API exposes selected schemas, chunking helpers, embedder factory,
guardrails, LLM clients, RAG pipeline, prompt registry, retrieval/reranking helpers, and
telemetry utilities through `genai_rag_engineering.__all__`.

Key public entry points include:

- `Document`, `Chunk`, `Query`, `RetrievedChunk`, `Prediction`, `EvalReport`
- `chunk_document`, `chunk_documents`, `chunk_size_stats`
- `FakeLLMClient`, `OpenAIClient`, `AnthropicClient`, `get_llm_client`
- `DenseRetriever`, `BM25Retriever`, `HybridRetriever`, `CrossEncoderReranker`
- `reciprocal_rank_fusion`, `hyde_transform`, `multi_query_rewrite`
- `InputGuardrail`, `OutputGuardrail`, `detect_pii`, `redact_pii`
- `RAGPipeline`
- `CostTracker`, `RollingLatencyTracker`, `Tracer`, `structured_log`

## Tests And Validation

Validation performed from `migration/exports/genai-rag-engineering`:

- `python -m pip install -e ".[dev]"`
- `ruff check .`
- `ruff format --check .`
- `mypy src tests`
- `pytest --cov=genai_rag_engineering --cov-report=term-missing`

Result:

- 6 tests passed.
- coverage: 65%.

Test coverage includes:

- overlapping document chunking and chunk statistics;
- PII detection/redaction;
- prompt-injection input guardrail;
- offline RAG pipeline execution with `FakeLLMClient` and citations;
- reciprocal-rank fusion;
- retrieval metrics for gold document hits.

## Runtime Modes

Default CI/runtime mode:

- deterministic `FakeLLMClient`;
- no API keys;
- no model downloads;
- small in-memory retriever test doubles.

Optional live mode:

- OpenAI and Anthropic wrappers are available behind optional dependencies and provider
  configuration.

Optional retrieval mode:

- heavier retrieval dependencies are separated behind the `retrieval` extra.

## Remaining Production Limitations

- Service/API code is represented as optional dependency metadata, but a maintained FastAPI
  service surface was not copied into the export.
- Live-provider integration is wrapper-level and not exercised in CI.
- Retrieval tests use small deterministic examples, not large vector-store regression tests.
- Notebook execution was not added to CI; notebooks were retained as portfolio examples.
- Generated vector artifacts are excluded and must be regenerated in any future standalone
  repository if needed.
- The export does not include deployment manifests because the monorepo deployment material
  is generic infrastructure rather than a focused GenAI service.

## History And Repository Status

This is a fresh snapshot export. No original portfolio source, notebook, artifact, or
documentation path was moved or deleted.
