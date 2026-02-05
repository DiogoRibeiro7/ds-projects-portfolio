"""
Mixture-of-Experts models with Gaussian experts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy import erf


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _normal_pdf(y: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    coeff = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return coeff * np.exp(-0.5 * ((y - mean) / sigma) ** 2)


def _normal_cdf(y: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    return 0.5 * (1.0 + erf(((y - mean) / sigma) / np.sqrt(2.0)))


@dataclass
class GaussianExpert:
    """Single Gaussian expert with linear mean function."""

    beta: np.ndarray
    sigma: float

    def mean(self, x: np.ndarray) -> np.ndarray:
        return x @ self.beta

    def pdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mean = self.mean(x)
        return _normal_pdf(y, mean, self.sigma)

    def cdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mean = self.mean(x)
        return _normal_cdf(y, mean, self.sigma)


class SoftmaxGatingNetwork:
    """Linear softmax gating network."""

    def __init__(self, n_features: int, n_experts: int, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(scale=0.1, size=(n_features, n_experts))

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights
        return _softmax(logits)

    def gradient_step(
        self, x: np.ndarray, target_probs: np.ndarray, lr: float
    ) -> np.ndarray:
        probs = self.probabilities(x)
        grad = x.T @ (probs - target_probs) / x.shape[0]
        self.weights -= lr * grad
        return probs


class GaussianMixtureOfExperts:
    """
    Mixture-of-Experts with Gaussian experts trained via EM-like updates.

    Args:
        n_experts: Number of experts in the mixture.
        max_iters: Maximum EM iterations.
        lr: Learning rate for gating updates.
        reg: Ridge regularization for expert regressions.
    """

    def __init__(
        self,
        n_experts: int,
        max_iters: int = 200,
        lr: float = 0.1,
        reg: float = 1e-3,
        seed: Optional[int] = None,
    ):
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        self.n_experts = n_experts
        self.max_iters = max_iters
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.gating: Optional[SoftmaxGatingNetwork] = None
        self.experts: list[GaussianExpert] = []

    def _prepare_features(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        bias = np.ones((x.shape[0], 1))
        return np.hstack([x, bias])

    def fit(self, x: np.ndarray, y: Iterable[float]) -> "GaussianMixtureOfExperts":
        x = self._prepare_features(x)
        y = np.asarray(list(y), dtype=float)
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("x and y must have compatible shapes")

        n_samples, n_features = x.shape
        rng = np.random.default_rng(self.seed)
        self.gating = SoftmaxGatingNetwork(n_features, self.n_experts, seed=self.seed)
        # Initialize experts
        betas = rng.normal(scale=0.1, size=(self.n_experts, n_features))
        sigmas = np.full(self.n_experts, np.std(y) + 1e-3)
        self.experts = [
            GaussianExpert(betas[k], sigmas[k]) for k in range(self.n_experts)
        ]

        for _ in range(self.max_iters):
            probs = self.gating.probabilities(x)
            pdfs = np.column_stack([exp.pdf(x, y) for exp in self.experts])
            weighted = probs * pdfs + 1e-12
            responsibilities = weighted / weighted.sum(axis=1, keepdims=True)

            # Update experts via weighted least squares
            for k in range(self.n_experts):
                weights = responsibilities[:, k] + 1e-8
                xw = x * weights[:, None]
                A = xw.T @ x + self.reg * np.eye(n_features)
                b_vec = xw.T @ y
                beta = np.linalg.solve(A, b_vec)
                mean = x @ beta
                var = np.average((y - mean) ** 2, weights=weights)
                sigma = max(math.sqrt(var + 1e-6), 1e-4)
                self.experts[k] = GaussianExpert(beta, sigma)

            self.gating.gradient_step(x, responsibilities, self.lr)

        return self

    def mixture_components(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.gating is None or not self.experts:
            raise ValueError("Model has not been fitted")
        x_feat = self._prepare_features(x)
        probs = self.gating.probabilities(x_feat)
        means = np.column_stack([exp.mean(x_feat) for exp in self.experts])
        sigmas = np.array([exp.sigma for exp in self.experts])
        return probs, means, sigmas

    def pdf(self, x: np.ndarray, y: Iterable[float]) -> np.ndarray:
        if self.gating is None or not self.experts:
            raise ValueError("Model has not been fitted")
        x_feat = self._prepare_features(x)
        y = np.asarray(list(y), dtype=float)
        probs = self.gating.probabilities(x_feat)
        pdfs = np.column_stack([exp.pdf(x_feat, y) for exp in self.experts])
        return np.sum(probs * pdfs, axis=1)

    def cdf(self, x: np.ndarray, y: Iterable[float]) -> np.ndarray:
        if self.gating is None or not self.experts:
            raise ValueError("Model has not been fitted")
        x_feat = self._prepare_features(x)
        y = np.asarray(list(y), dtype=float)
        probs = self.gating.probabilities(x_feat)
        cdfs = np.column_stack([exp.cdf(x_feat, y) for exp in self.experts])
        return np.sum(probs * cdfs, axis=1)

    def negative_log_likelihood(self, x: np.ndarray, y: Iterable[float]) -> float:
        if self.gating is None or not self.experts:
            raise ValueError("Model has not been fitted")
        pdf_vals = self.pdf(x, y)
        return float(-np.mean(np.log(pdf_vals + 1e-12)))
