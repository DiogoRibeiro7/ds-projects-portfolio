# GenAI RAG Engineering

Standalone-ready export of the portfolio's RAG and LLM engineering utilities.

This export is copy-first: the original `src/genai` package and notebooks remain in
`ds-projects-portfolio`. The default test path uses `FakeLLMClient`, so CI runs offline
without API keys or model downloads.

## Included Scope

- Document schemas and chunking
- Prompt registry and RAG orchestration
- Deterministic fake LLM client
- Optional OpenAI and Anthropic client wrappers
- Retrieval and reranking primitives
- Guardrails, PII redaction, and prompt-injection checks
- RAG evaluation metrics and judge helpers
- Cost, latency, and trace telemetry
- Notebook examples copied under `notebooks/`

## Notebook Examples

- `notebooks/genai_rag_pipeline.ipynb`
- `notebooks/llm_rag_evaluation.ipynb`
- `notebooks/genai_service_delivery.ipynb`
- `notebooks/genai_dataops_vector_platform.ipynb`

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src tests
pytest --cov=genai_rag_engineering --cov-report=term-missing
```

Live provider clients and heavyweight retrieval extras are optional:

```bash
python -m pip install -e ".[live,retrieval]"
```
