"""Prompt registry with immutable versioning.

Every prompt used by the pipeline or eval lives here keyed by `(id, version)`. Changes
bump the version string; the prior version stays accessible for A/B comparisons and
regression eval. Each `Prediction` carries its `prompt_id` and `prompt_version` so
rollbacks are traceable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    system: str
    user_template: str
    notes: str = ""

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """Return (system, user) — the system string unchanged + user_template.format(**kwargs)."""
        return self.system, self.user_template.format(**kwargs)


@dataclass
class PromptRegistry:
    """Simple in-process prompt registry. Add once, read anywhere."""
    _store: dict[tuple[str, str], Prompt] = field(default_factory=dict)
    _latest: dict[str, str] = field(default_factory=dict)

    def register(self, prompt: Prompt) -> None:
        key = (prompt.id, prompt.version)
        if key in self._store:
            raise ValueError(f"Prompt {key} already registered — bump version")
        self._store[key] = prompt
        self._latest[prompt.id] = prompt.version

    def get(self, prompt_id: str, version: str | None = None) -> Prompt:
        v = version or self._latest[prompt_id]
        return self._store[(prompt_id, v)]

    def list_ids(self) -> list[str]:
        return sorted(self._latest.keys())

    def versions(self, prompt_id: str) -> list[str]:
        return sorted(v for (i, v) in self._store if i == prompt_id)


# ----------------------------------------------------------------------------
# Seed the registry with the baseline RAG + judge prompts
# ----------------------------------------------------------------------------

REGISTRY = PromptRegistry()

REGISTRY.register(Prompt(
    id="rag.answer",
    version="v1",
    system=(
        "You are a precise assistant for a regulated-industry knowledge base. "
        "Answer ONLY using the provided context. If the context is insufficient, "
        "say so explicitly. Cite the context blocks you used by their bracketed IDs "
        "(e.g. [KB-0001]). Do NOT invent citations. Keep the answer under 180 words."
    ),
    user_template=(
        "### CONTEXT\n"
        "{context}\n\n"
        "### QUESTION\n"
        "{question}\n\n"
        "### INSTRUCTIONS\n"
        "Respond in 3-6 sentences. End with a single line of the form:\n"
        "Citations: [ID1], [ID2], ...\n"
        "If you cannot answer from the context, reply with:\n"
        "'I cannot answer from the provided context.' and no citations."
    ),
    notes="v1: baseline single-turn, citation-suffixed. Faithfulness-first, short.",
))

REGISTRY.register(Prompt(
    id="judge.faithfulness",
    version="v1",
    system=(
        "You are an impartial judge of factual faithfulness. Given a question, the "
        "retrieved context, and a candidate answer, decide whether every factual claim "
        "in the answer is supported by the context. Return strict JSON only."
    ),
    user_template=(
        "### QUESTION\n{question}\n\n"
        "### CONTEXT\n{context}\n\n"
        "### ANSWER\n{answer}\n\n"
        "### TASK\n"
        "Evaluate the answer's faithfulness to the context. Produce JSON with keys:\n"
        "  label: one of 'faithful', 'partial', 'unfaithful'\n"
        "  score: 1.0 (faithful) / 0.5 (partial) / 0.0 (unfaithful)\n"
        "  reasoning: 1-2 sentences\n"
        "Output only the JSON object."
    ),
    notes="v1: 3-class label + score, strict JSON.",
))

REGISTRY.register(Prompt(
    id="judge.answer_relevance",
    version="v1",
    system=(
        "You are an impartial judge. Given a question and a candidate answer, decide "
        "how directly the answer addresses the question. Return strict JSON only."
    ),
    user_template=(
        "### QUESTION\n{question}\n\n"
        "### ANSWER\n{answer}\n\n"
        "### TASK\n"
        "Rate how directly the answer addresses the question on a 0-1 scale:\n"
        "  1.0  the answer directly addresses the question\n"
        "  0.5  the answer is partially on-topic\n"
        "  0.0  the answer is off-topic or a refusal\n"
        "Return JSON with keys: score (float), reasoning (1 sentence)."
    ),
))

REGISTRY.register(Prompt(
    id="judge.context_precision",
    version="v1",
    system=(
        "You are an impartial judge. Given a question and a single retrieved passage, "
        "decide whether the passage is relevant to the question. Return strict JSON."
    ),
    user_template=(
        "### QUESTION\n{question}\n\n"
        "### PASSAGE\n{passage}\n\n"
        "Return JSON: {{\"relevant\": true/false, \"reasoning\": \"<1 sentence>\"}}"
    ),
))
