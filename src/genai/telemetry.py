"""Minimal, dependency-free tracing + cost bookkeeping.

Three components:

- `Tracer` — a context manager that accumulates spans (name, start, end, attributes)
  into an in-memory list. A tracer can be exported as a list of dicts and embedded
  in `Prediction.trace` so every request carries its own causal chain.
- `CostTracker` — token × provider-rate → USD. Pricing table is a simple dict and
  can be overridden per-notebook (e.g., to toggle between current and historical
  pricing for sensitivity analysis).
- `RollingLatencyTracker` — sliding-window P50/P95 computation for online SLA checks.

These are intentionally thin so a notebook can show OpenTelemetry / LangFuse wiring
at a conceptual level without adding those as runtime deps.
"""
from __future__ import annotations

import contextlib
import json
import statistics
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
        }


class Tracer:
    """Context-manager-based span tracker.

    Usage:
        tracer = Tracer()
        with tracer.span("retrieve", {"k": 10}) as span:
            ...do work...
            span["attributes"]["n_hits"] = 7

    The span dict is mutable so the caller can add attributes as work proceeds.
    """

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self._spans: list[Span] = []
        self._origin_ns = time.monotonic_ns()

    def _now_ms(self) -> float:
        return (time.monotonic_ns() - self._origin_ns) / 1e6

    @contextlib.contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        mutable = {"name": name, "attributes": dict(attributes or {})}
        start = self._now_ms()
        try:
            yield mutable
        finally:
            end = self._now_ms()
            self._spans.append(Span(
                name=mutable["name"],
                start_ms=start,
                end_ms=end,
                attributes=mutable["attributes"],
            ))

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._spans]

    def total_ms(self) -> float:
        if not self._spans:
            return 0.0
        return max(s.end_ms for s in self._spans) - min(s.start_ms for s in self._spans)


# ----------------------------------------------------------------------------
# Cost
# ----------------------------------------------------------------------------

# Prices are USD per 1M tokens. Anthropic & OpenAI 2026 published rates (representative).
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1m, output_per_1m)
    "claude-opus-4-7":        (15.00, 75.00),
    "claude-sonnet-4-6":      (3.00,  15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gpt-4o":                 (2.50,  10.00),
    "gpt-4o-mini":            (0.15,  0.60),
    "fake-default":           (0.00,  0.00),
    "fake-sonnet":            (0.00,  0.00),
}


@dataclass
class CostTracker:
    pricing: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICING))
    total_usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)
    by_tenant: dict[str, float] = field(default_factory=dict)

    def record(self, *, model: str, prompt_tokens: int, completion_tokens: int,
               tenant_id: str = "default") -> float:
        rates = self.pricing.get(model, (0.0, 0.0))
        usd = (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000
        self.total_usd += usd
        self.by_model[model] = self.by_model.get(model, 0.0) + usd
        self.by_tenant[tenant_id] = self.by_tenant.get(tenant_id, 0.0) + usd
        return usd


# ----------------------------------------------------------------------------
# Rolling latency (P50/P95 for online SLA checks)
# ----------------------------------------------------------------------------

class RollingLatencyTracker:
    def __init__(self, window: int = 500):
        self._window = window
        self._buf: deque[float] = deque(maxlen=window)

    def record(self, latency_ms: float) -> None:
        self._buf.append(float(latency_ms))

    def p(self, q: float) -> float:
        if not self._buf:
            return 0.0
        vals = sorted(self._buf)
        idx = min(len(vals) - 1, max(0, int(q * len(vals)) - 1))
        return vals[idx]

    def summary(self) -> dict[str, float]:
        if not self._buf:
            return {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
        arr = list(self._buf)
        return {
            "n": len(arr),
            "mean": statistics.fmean(arr),
            "p50": self.p(0.50),
            "p95": self.p(0.95),
        }


# ----------------------------------------------------------------------------
# Structured JSON logger
# ----------------------------------------------------------------------------

def structured_log(level: str, event: str, **fields: Any) -> str:
    """Serialise a log line to one-line JSON; returns the string for testability.

    Use with `print(structured_log(...))` in notebooks or `logger.info(structured_log(...))`
    in production apps. Kept deliberately simple to avoid pulling in structlog.
    """
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level.upper(),
        "event": event,
        **fields,
    }
    return json.dumps(payload, default=str, separators=(",", ":"))
