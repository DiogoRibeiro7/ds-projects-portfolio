"""Lightweight input / output guardrails for the RAG pipeline.

Two concerns, four detectors:

- **PII** at ingress: email, phone, SSN, credit-card, IBAN, IP — regex-based. Findings
  are redacted in-place; the original is never stored in predictions.jsonl.
- **Prompt injection** at ingress: heuristic patterns for override / jailbreak / role-swap
  attempts (Perez & Ribeiro 2022-style). Blocks and logs the finding; pipeline returns a
  canned refusal.
- **Hallucination** at egress: a claim is unsupported if none of its content-words overlap
  with any retrieved chunk. Surfaced as a warning, not a hard block.
- **PII leakage** at egress: the same PII regex run on the generated answer, flagging if
  the model has emitted a PII token that does not appear in the retrieved context.

`presidio-analyzer` can be swapped in with minimal effort; we use regex here so the
notebook runs offline with no spaCy download.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import GuardrailFinding, GuardrailResult

# ----------------------------------------------------------------------------
# PII detectors (English-only, conservative — intentionally high-precision)
# ----------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b")),
    ("ssn",   re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("iban",  re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("ip",    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def detect_pii(text: str) -> list[GuardrailFinding]:
    findings: list[GuardrailFinding] = []
    for name, pat in _PII_PATTERNS:
        for m in pat.finditer(text):
            matched = m.group(0)
            # Credit-card pattern can false-positive on long digit runs; cheap Luhn-less sanity check
            if name == "credit_card" and len(re.sub(r"\D", "", matched)) < 13:
                continue
            findings.append(GuardrailFinding(
                name=f"pii.{name}",
                severity="warn",
                matched=matched,
                reason=f"matched {name} regex",
            ))
    return findings


def redact_pii(text: str) -> tuple[str, list[GuardrailFinding]]:
    """Return a redacted copy of `text` and the list of findings."""
    findings = detect_pii(text)
    redacted = text
    for f in findings:
        redacted = redacted.replace(f.matched, f"[REDACTED_{f.name.upper()}]")
    return redacted, findings


# ----------------------------------------------------------------------------
# Prompt-injection heuristics
# ----------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("override", re.compile(
        r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier)\b",
        re.IGNORECASE,
    )),
    ("system_reveal", re.compile(
        r"\b(reveal|show|print|output)\b.{0,40}\b(system\s+prompt|instructions|rules)\b",
        re.IGNORECASE,
    )),
    ("role_swap", re.compile(
        r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as)\b",
        re.IGNORECASE,
    )),
    ("exfil", re.compile(
        r"\b(send|email|post|upload|leak)\b.{0,40}\b(to|at)\b\s+\S+@\S+",
        re.IGNORECASE,
    )),
    ("harm_intent", re.compile(
        r"\b(how\s+(?:to|do\s+i|can\s+i)\s+(?:make|build|synthes[iy]ze|manufacture))\b"
        r".{0,40}\b(bomb|explosive|weapon|malware|virus|gun)\b",
        re.IGNORECASE,
    )),
]


def detect_prompt_injection(text: str) -> list[GuardrailFinding]:
    out: list[GuardrailFinding] = []
    for name, pat in _INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            out.append(GuardrailFinding(
                name=f"injection.{name}",
                severity="block",
                matched=m.group(0),
                reason=f"heuristic match for {name}",
            ))
    return out


# ----------------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------------

@dataclass
class InputGuardrail:
    """Ingress check: redact PII, block prompt-injection."""

    def check(self, text: str) -> GuardrailResult:
        redacted, pii = redact_pii(text)
        injection = detect_prompt_injection(redacted)
        findings = pii + injection
        return GuardrailResult(
            blocked=any(f.severity == "block" for f in findings),
            findings=findings,
            redacted_text=redacted,
        )


@dataclass
class OutputGuardrail:
    """Egress check: flag unsupported claims and PII leakage.

    Unsupported-claim check: tokenise the answer into content words (len > 3,
    not a stopword-ish list) and flag those absent from the union of retrieved
    chunk texts. If the unsupported-content ratio exceeds `hallucination_threshold`,
    a `warn` finding is emitted.
    """
    hallucination_threshold: float = 0.50

    def check(self, answer_text: str, retrieved_texts: list[str]) -> GuardrailResult:
        findings: list[GuardrailFinding] = []

        # PII leakage at egress
        pii = detect_pii(answer_text)
        for f in pii:
            # Only flag as leakage if PII is not also in the retrieved context (which could be
            # a legitimate reason to surface it, e.g. a policy document quoting a contact line).
            if not any(f.matched in t for t in retrieved_texts):
                findings.append(GuardrailFinding(
                    name="pii_leak." + f.name.split(".", 1)[1],
                    severity="warn",
                    matched=f.matched,
                    reason="PII in answer not present in retrieved context",
                ))

        # Hallucination / unsupported-content check
        context_words = set()
        for t in retrieved_texts:
            context_words.update(re.findall(r"[A-Za-z0-9]+", t.lower()))
        content_words = [
            w for w in re.findall(r"[A-Za-z0-9]+", answer_text.lower()) if len(w) > 3
        ]
        if content_words:
            unsupported = [w for w in content_words if w not in context_words]
            ratio = len(unsupported) / len(content_words)
            if ratio > self.hallucination_threshold:
                findings.append(GuardrailFinding(
                    name="hallucination.unsupported_content",
                    severity="warn",
                    matched=",".join(unsupported[:10]),
                    reason=f"{ratio:.0%} of content words not found in retrieved context",
                ))

        return GuardrailResult(
            blocked=False,
            findings=findings,
            redacted_text=answer_text,
        )


REFUSAL_MESSAGE = (
    "I can't answer that request. It may contain an instruction-override pattern "
    "or a harmful-intent signal flagged by the input guardrail."
)
