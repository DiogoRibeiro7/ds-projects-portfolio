# GenAI RAG Engineering Export Report

## Decision

Create a standalone-ready GenAI/RAG package as a copy-first export.
No source files, notebooks, artifacts, or portfolio assets were moved or deleted from
`ds-projects-portfolio`.

## Export Location

`migration/exports/genai-rag-engineering`

## Copied From Portfolio

- `src/genai/*` to `src/genai_rag_engineering/*`
- `notebooks/genai_rag_pipeline.ipynb`
- `notebooks/llm_rag_evaluation.ipynb`
- `notebooks/genai_service_delivery.ipynb`
- `notebooks/genai_dataops_vector_platform.ipynb`

## Kept In Portfolio

- Original `src/genai`
- Original GenAI notebooks under `notebooks/`
- Generated artifacts under `artifacts/genai/`
- Portfolio tests and docs

## Validation Commands

Run from the export directory:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src tests
pytest --cov=genai_rag_engineering --cov-report=term-missing
```
