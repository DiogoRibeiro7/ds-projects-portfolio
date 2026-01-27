"""
Monotone quantile models with cumulative softplus heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class MonotoneQuantileHead:
    """
    Transform raw network outputs into monotone quantiles.

    The head expects a tensor with shape ``(n_samples, n_quantiles + 1)``
    where the first column is the base prediction and the remaining columns
    are unconstrained increments. Quantiles are computed as::

        q_j = base + sum_{k<=j} softplus(delta_k)

    which ensures positivity of increments and therefore non-crossing quantiles.
    """

    quantile_levels: Sequence[float]

    def __post_init__(self) -> None:
        levels = np.asarray(self.quantile_levels, dtype=float)
        if levels.ndim != 1:
            raise ValueError("quantile_levels must be 1D")
        if np.any((levels <= 0) | (levels >= 1)):
            raise ValueError("quantiles must be in (0, 1)")
        if not np.all(np.diff(levels) > 0):
            raise ValueError("quantile_levels must be strictly increasing")
        self._levels = levels

    @property
    def n_quantiles(self) -> int:
        return len(self._levels)

    def __call__(self, raw_outputs: np.ndarray) -> np.ndarray:
        """Convert raw outputs to monotone quantiles."""
        raw = np.asarray(raw_outputs, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != self.n_quantiles + 1:
            raise ValueError(
                "raw_outputs must have shape (n_samples, n_quantiles + 1)"
            )
        base = raw[:, [0]]
        deltas = raw[:, 1:]
        increments = _softplus(deltas)
        cumulative = np.cumsum(increments, axis=1)
        quantiles = base + cumulative
        return quantiles


def pinball_loss(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantile_levels: Sequence[float],
) -> float:
    """Compute the multi-quantile pinball loss."""
    y = np.asarray(y_true, dtype=float)
    preds = np.asarray(quantile_preds, dtype=float)
    taus = np.asarray(list(quantile_levels), dtype=float)
    if preds.ndim == 1:
        preds = np.broadcast_to(preds, (y.shape[0], preds.shape[0]))
    if preds.shape != (y.shape[0], taus.shape[0]):
        raise ValueError("quantile_preds must have shape (n_samples, n_quantiles)")
    diff = y[:, None] - preds
    loss = np.maximum(taus * diff, (taus - 1.0) * diff)
    return float(loss.mean())


def fit_constant_monotone_quantiles(
    y: Iterable[float],
    quantile_levels: Sequence[float],
    epochs: int = 2000,
    lr: float = 0.05,
) -> dict[str, np.ndarray]:
    """
    Fit constant (featureless) monotone quantiles via gradient descent.
    """
    data = np.asarray(list(y), dtype=float)
    if data.ndim != 1:
        raise ValueError("y must be 1D")
    levels = np.asarray(list(quantile_levels), dtype=float)
    head = MonotoneQuantileHead(levels)
    base = np.median(data)
    s = np.zeros(head.n_quantiles)
    for _ in range(epochs):
        increments = _softplus(s)
        quantiles = base + np.cumsum(increments)
        indicators = (data[:, None] < quantiles[None, :]).astype(float)
        grad_q = indicators.mean(axis=0) - levels
        grad_base = grad_q.sum()
        softplus_grad = _sigmoid(s)
        future = np.cumsum(grad_q[::-1])[::-1]
        grad_s = softplus_grad * future
        base -= lr * grad_base
        s -= lr * grad_s
    final_increments = _softplus(s)
    final_quantiles = base + np.cumsum(final_increments)
    return {
        "base": np.array([base]),
        "raw_increments": s.copy(),
        "quantiles": final_quantiles,
    }
