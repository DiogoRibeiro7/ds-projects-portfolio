"""LLM client abstraction.

The three notebooks default to a deterministic `FakeLLMClient` so they run offline,
reproducibly, and without API credentials. Toggling a `USE_LIVE_API` flag at the top
of each notebook swaps in the real Anthropic or OpenAI client with identical call
signatures and response shapes.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any, Protocol, cast

from .schemas import LLMMessage, LLMResponse


class LLMClient(Protocol):
    """Protocol for all LLM clients."""

    model: str
    provider: str

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


# ----------------------------------------------------------------------------
# Real clients (thin, single-responsibility wrappers around the SDK response)
# ----------------------------------------------------------------------------


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None
    ):
        import anthropic  # local import: keep module import cheap

        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        system = None
        user_msgs = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                user_msgs.append({"role": m.role, "content": m.content})
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_msgs,
        }
        if system is not None:
            kwargs["system"] = system
        t0 = time.monotonic()
        resp = self._client.messages.create(**kwargs)
        latency = (time.monotonic() - t0) * 1000
        text = "".join(
            getattr(b, "text", "")
            for b in resp.content
            if getattr(b, "type", "") == "text"
        )
        return LLMResponse(
            text=text,
            model=self.model,
            provider="anthropic",
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            latency_ms=latency,
            stop_reason=resp.stop_reason or "",
        )


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        import openai

        self.model = model
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        t0 = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=cast(
                Any,
                [{"role": m.role, "content": m.content} for m in messages],
            ),
        )
        latency = (time.monotonic() - t0) * 1000
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=self.model,
            provider="openai",
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            latency_ms=latency,
            stop_reason=choice.finish_reason or "",
        )


# ----------------------------------------------------------------------------
# Deterministic offline fake — the default for reproducible notebook runs.
# ----------------------------------------------------------------------------

_CITE_PATTERN = re.compile(r"\[([A-Za-z0-9_\-#]+)\]\s*(.*)")
_JUDGE_HINT = re.compile(r"score\s*[:=]\s*\{?", re.IGNORECASE)
_JUDGE_SCORE = re.compile(
    r"rate this as\s+(faithful|partial|unfaithful)", re.IGNORECASE
)


class FakeLLMClient:
    """Deterministic offline stand-in for a real LLM.

    For **generation** prompts containing context blocks of the form `[DOC-ID] <text>`,
    returns an abstractive-looking answer that cites the first block by ID and quotes
    its opening clause. For **judging** prompts (see `src/genai/evals.py`), recognises
    a rubric signal and returns a deterministic JSON verdict. Latency and token counts
    are seeded so cost / latency dashboards render sensibly even offline.
    """

    provider = "fake"

    def __init__(self, model: str = "fake-sonnet", seed: int = 17):
        self.model = model
        self.seed = seed

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        system = next((m.content for m in messages if m.role == "system"), "")
        full_prompt = system + "\n\n" + last_user
        prompt_digest = hashlib.md5(full_prompt.encode("utf-8", "ignore")).hexdigest()[
            :8
        ]

        text = self._respond(full_prompt, prompt_digest)
        token_in = max(1, len(full_prompt.split()))
        token_out = max(1, len(text.split()))

        # Reproducible pseudo-latency; small jitter keyed on prompt hash so different prompts
        # do not collapse to identical latency while still being deterministic.
        base = 110.0
        jitter = int(prompt_digest, 16) % 60
        latency = base + jitter

        return LLMResponse(
            text=text,
            model=self.model,
            provider="fake",
            prompt_tokens=token_in,
            completion_tokens=token_out,
            latency_ms=latency,
            stop_reason="stop",
        )

    def _respond(self, prompt: str, digest: str) -> str:
        # Dispatch on the judge-prompt shape (dimension-specific JSON schemas)
        lower = prompt.lower()
        if "faithfulness" in lower and "### context" in lower:
            return self._respond_judge_faithfulness(prompt)
        if "how directly the answer addresses" in lower:
            return self._respond_judge_answer_relevance(prompt)
        if "whether the passage is relevant" in lower:
            return self._respond_judge_context_precision(prompt)
        # Detect HyDE-style prompts
        if "hypothetical passage" in lower:
            return self._respond_hyde(prompt)
        # Detect multi-query rewriter
        if "reformulations of the following query" in lower:
            return self._respond_multi_query(prompt)
        # Default: citation-style generation from retrieved context
        return self._respond_generation(prompt)

    def _respond_generation(self, prompt: str) -> str:
        # Look for lines that match "[DOC-ID] text"
        hits = []
        for line in prompt.splitlines():
            m = _CITE_PATTERN.match(line.strip())
            if m:
                hits.append((m.group(1), m.group(2).strip()))
        if not hits:
            return (
                "I don't have enough grounded context to answer that confidently. "
                "Please rephrase or provide source material."
            )
        doc_id, body = hits[0]
        # Use the first 200 characters of the first hit as the main quote
        quote = body[:200].rstrip(" ,.;:")
        summary = quote + "." if not quote.endswith(".") else quote
        cite_list = ", ".join(f"[{d}]" for d, _ in hits[:3])
        return (
            f"{summary}\n\nCitations: {cite_list}.\n\n"
            f"_Generated offline by FakeLLMClient — flip USE_LIVE_API to call a real model._"
        )

    def _respond_judge_faithfulness(self, prompt: str) -> str:
        ctx = _extract_section(prompt, "### CONTEXT")
        ans = _extract_section(prompt, "### ANSWER")
        ctx_tokens = set(_word_tokens(ctx))
        ans_tokens = [w for w in _word_tokens(ans) if len(w) > 3]
        if not ans_tokens:
            score, label = 0.0, "unfaithful"
        else:
            overlap = sum(1 for w in ans_tokens if w in ctx_tokens) / len(ans_tokens)
            if overlap > 0.60:
                score, label = 1.0, "faithful"
            elif overlap > 0.30:
                score, label = 0.5, "partial"
            else:
                score, label = 0.0, "unfaithful"
        return (
            '{"label": "' + label + '", '
            f'"score": {score}, '
            '"reasoning": "overlap heuristic (offline fake judge)"}'
        )

    def _respond_judge_answer_relevance(self, prompt: str) -> str:
        q = _extract_section(prompt, "### QUESTION")
        a = _extract_section(prompt, "### ANSWER")
        q_tokens = set(t for t in _word_tokens(q) if len(t) > 3)
        a_tokens = [t for t in _word_tokens(a) if len(t) > 3]
        if not a_tokens or not q_tokens:
            score = 0.0
        else:
            overlap = sum(1 for t in a_tokens if t in q_tokens) / len(a_tokens)
            # Answers tend to be longer than questions; a modest overlap signals on-topic.
            score = 1.0 if overlap > 0.12 else (0.5 if overlap > 0.04 else 0.0)
        return f'{{"score": {score}, "reasoning": "question-answer overlap heuristic"}}'

    def _respond_judge_context_precision(self, prompt: str) -> str:
        q = _extract_section(prompt, "### QUESTION")
        p = _extract_section(prompt, "### PASSAGE")
        q_tokens = set(t for t in _word_tokens(q) if len(t) > 3)
        p_tokens = set(t for t in _word_tokens(p) if len(t) > 3)
        if not q_tokens or not p_tokens:
            rel = False
        else:
            overlap = len(q_tokens & p_tokens) / len(q_tokens)
            rel = overlap > 0.20
        return (
            '{"relevant": true, "reasoning": "token overlap > 0.20"}'
            if rel
            else '{"relevant": false, "reasoning": "insufficient token overlap"}'
        )

    def _respond_hyde(self, prompt: str) -> str:
        # Extract the question after "Question:"
        m = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt)
        q = m.group(1).strip() if m else "the question"
        return (
            f"The answer addresses {q.lower()} by referencing the relevant policy document. "
            f"The applicable terms describe the handling procedure and the conditions that "
            f"trigger it, with specific references to the coverage and exclusion clauses."
        )

    def _respond_multi_query(self, prompt: str) -> str:
        m = re.search(r"Original query:\s*(.+?)(?:\n|$)", prompt)
        q = m.group(1).strip() if m else ""
        variants = [
            q,
            q.replace("how do", "what is the process to").replace("what is", "explain"),
            f"Rephrase of: {q}",
        ]
        # JSON list so parsing is clean
        return "[" + ", ".join(f'"{v}"' for v in variants) + "]"


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _extract_section(prompt: str, header: str) -> str:
    """Return the text between `header` and the next `###` header (or end of string)."""
    idx = prompt.find(header)
    if idx < 0:
        return ""
    rest = prompt[idx + len(header) :]
    end = rest.find("###")
    return rest[:end] if end >= 0 else rest


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


def get_llm_client(
    provider: str = "fake",
    model: str | None = None,
    *,
    use_live: bool = False,
) -> LLMClient:
    """Construct an LLM client.

    `use_live=False` returns a FakeLLMClient regardless of `provider` so that notebooks
    produce identical outputs in CI / offline runs. Set `use_live=True` + export the
    appropriate `{ANTHROPIC,OPENAI}_API_KEY` to exercise a real model.
    """
    if not use_live or provider == "fake":
        return FakeLLMClient(model=model or "fake-default")
    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-haiku-4-5-20251001")
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini")
    raise ValueError(f"Unknown LLM provider: {provider!r}")
