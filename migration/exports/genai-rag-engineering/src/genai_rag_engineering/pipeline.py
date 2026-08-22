"""RAG pipeline orchestrator.

Composes guardrails, retrieval, reranking, prompt rendering, LLM generation, output
guardrails, and cost / latency accounting into a single `RAGPipeline.run()` call that
returns a fully-populated `Prediction`.

Designed so the same pipeline object is reusable from:
  - `genai_rag_pipeline.ipynb` — loops it across a dataset and writes predictions.jsonl
  - `genai_service_delivery.ipynb` — serves it from a FastAPI endpoint
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .guardrails import REFUSAL_MESSAGE, InputGuardrail, OutputGuardrail
from .llm import LLMClient
from .prompts import REGISTRY as PROMPT_REGISTRY
from .prompts import Prompt
from .schemas import (
    Citation,
    GeneratedAnswer,
    GuardrailResult,
    LLMMessage,
    LLMResponse,
    Prediction,
    Query,
    RetrievedChunk,
)
from .telemetry import CostTracker, Tracer


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]: ...


_CITE_ID_RE = re.compile(r"\[([A-Za-z0-9][\w\-#]*)\]")


def _parse_citations(answer_text: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
    """Extract IDs that appear in brackets in the answer; map them back to retrieved chunks."""
    seen: set[str] = set()
    citations: list[Citation] = []
    retrieved_by_doc: dict[str, RetrievedChunk] = {}
    retrieved_by_chunk: dict[str, RetrievedChunk] = {}
    for r in retrieved:
        retrieved_by_doc.setdefault(r.doc_id, r)
        retrieved_by_chunk.setdefault(r.chunk_id, r)
    for match in _CITE_ID_RE.finditer(answer_text):
        ref = match.group(1)
        if ref in seen:
            continue
        seen.add(ref)
        hit = retrieved_by_chunk.get(ref) or retrieved_by_doc.get(ref)
        if hit is None:
            # Citation to something outside the retrieved set — still record it so the eval
            # can flag it as a potential hallucinated citation.
            citations.append(Citation(chunk_id=ref, doc_id=ref, quote=""))
        else:
            citations.append(Citation(chunk_id=hit.chunk_id, doc_id=hit.doc_id, quote=hit.text[:180]))
    return citations


def _render_context(retrieved: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as bracketed blocks the model can cite."""
    blocks = []
    for r in retrieved:
        blocks.append(f"[{r.doc_id}] {r.text.strip()}")
    return "\n\n".join(blocks)


@dataclass
class RAGPipeline:
    retriever: Retriever
    llm: LLMClient
    reranker: Reranker | None = None
    retrieve_k: int = 10
    top_k_after_rerank: int = 5
    prompt_id: str = "rag.answer"
    prompt_version: str = "v1"
    input_guardrail: InputGuardrail = field(default_factory=InputGuardrail)
    output_guardrail: OutputGuardrail = field(default_factory=OutputGuardrail)
    cost_tracker: CostTracker | None = None
    retriever_label: str = "dense"

    def _prompt(self) -> Prompt:
        return PROMPT_REGISTRY.get(self.prompt_id, self.prompt_version)

    def run(self, query: Query) -> Prediction:
        tracer = Tracer()

        # ---- 1. Input guardrail ----
        with tracer.span("guardrail.input") as span:
            g_in = self.input_guardrail.check(query.text)
            span["attributes"]["blocked"] = g_in.blocked
            span["attributes"]["n_findings"] = len(g_in.findings)
        if g_in.blocked:
            return self._refusal(query, g_in, tracer)

        safe_text = g_in.redacted_text or query.text

        # ---- 2. Retrieve ----
        with tracer.span("retrieve", {"k": self.retrieve_k}) as span:
            raw_hits = self.retriever.retrieve(safe_text, k=self.retrieve_k)
            span["attributes"]["n_hits"] = len(raw_hits)

        # ---- 3. Rerank (optional) ----
        if self.reranker is not None and raw_hits:
            with tracer.span("rerank", {"top_k": self.top_k_after_rerank}) as span:
                ranked = self.reranker.rerank(safe_text, raw_hits, top_k=self.top_k_after_rerank)
                span["attributes"]["n_after"] = len(ranked)
        else:
            ranked = raw_hits[: self.top_k_after_rerank]

        # ---- 4. Render prompt + call LLM ----
        prompt = self._prompt()
        context = _render_context(ranked)
        system, user = prompt.render(context=context, question=safe_text)
        messages = [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]

        with tracer.span("llm.complete", {"model": self.llm.model, "provider": self.llm.provider}) as span:
            llm_resp: LLMResponse = self.llm.complete(messages, max_tokens=400, temperature=0.0)
            span["attributes"].update(
                {
                    "prompt_tokens": llm_resp.prompt_tokens,
                    "completion_tokens": llm_resp.completion_tokens,
                    "latency_ms": llm_resp.latency_ms,
                }
            )

        # ---- 5. Parse citations, detect refusal ----
        refused = llm_resp.text.strip().lower().startswith("i cannot answer")
        answer = GeneratedAnswer(
            answer=llm_resp.text,
            citations=_parse_citations(llm_resp.text, ranked),
            refused=refused,
            refusal_reason="insufficient_context" if refused else "",
        )

        # ---- 6. Output guardrail ----
        with tracer.span("guardrail.output") as span:
            g_out = self.output_guardrail.check(llm_resp.text, [r.text for r in ranked])
            span["attributes"]["n_findings"] = len(g_out.findings)

        # ---- 7. Cost ----
        cost = 0.0
        if self.cost_tracker is not None:
            cost = self.cost_tracker.record(
                model=llm_resp.model,
                prompt_tokens=llm_resp.prompt_tokens,
                completion_tokens=llm_resp.completion_tokens,
                tenant_id=query.tenant_id,
            )

        return Prediction(
            request_id=query.request_id,
            tenant_id=query.tenant_id,
            query=query.text,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            retriever=self.retriever_label,
            retrieved=ranked,
            answer=answer,
            llm=llm_resp,
            guardrails_input=g_in,
            guardrails_output=g_out,
            cost_usd=cost,
            latency_ms_total=tracer.total_ms(),
            trace=tracer.to_list(),
        )

    # ------------------------------------------------------------------

    def _refusal(self, query: Query, g_in: GuardrailResult, tracer: Tracer) -> Prediction:
        ans = GeneratedAnswer(
            answer=REFUSAL_MESSAGE,
            citations=[],
            refused=True,
            refusal_reason="input_guardrail_blocked",
        )
        # Use a zero-cost, zero-latency LLMResponse-like record to keep downstream shapes stable.
        llm_resp = LLMResponse(
            text=REFUSAL_MESSAGE,
            model=self.llm.model,
            provider=self.llm.provider,  # type: ignore[arg-type]
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
            stop_reason="refused",
        )
        return Prediction(
            request_id=query.request_id,
            tenant_id=query.tenant_id,
            query=query.text,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            retriever=self.retriever_label,
            retrieved=[],
            answer=ans,
            llm=llm_resp,
            guardrails_input=g_in,
            guardrails_output=GuardrailResult(),
            cost_usd=0.0,
            latency_ms_total=tracer.total_ms(),
            trace=tracer.to_list(),
        )
