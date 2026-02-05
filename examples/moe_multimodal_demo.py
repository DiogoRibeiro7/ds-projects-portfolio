"""Mixture-of-Experts demo on a multimodal dataset."""

from __future__ import annotations

import numpy as np

from src.statistics.moe import GaussianMixtureOfExperts


def make_data(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=n)
    y = np.where(
        x < 0,
        rng.normal(loc=-2 + 0.5 * x, scale=0.5, size=n),
        rng.normal(loc=2 + 0.5 * x, scale=0.4, size=n),
    )
    return x[:, None], y


def main() -> None:
    x, y = make_data()
    moe = GaussianMixtureOfExperts(n_experts=2, max_iters=300, lr=0.2)
    moe.fit(x, y)
    single = GaussianMixtureOfExperts(n_experts=1, max_iters=300)
    single.fit(x, y)

    nll_moe = moe.negative_log_likelihood(x, y)
    nll_single = single.negative_log_likelihood(x, y)

    print("=== Mixture-of-Experts Demo ===")
    print(f"MoE (K=2)   NLL: {nll_moe:.3f}")
    print(f"Single expert NLL: {nll_single:.3f}")
    if nll_moe < nll_single:
        print("✅ MoE captures multimodality better than single expert.")
    else:
        print("⚠️ MoE did not improve; consider adjusting hyperparameters.")


if __name__ == "__main__":
    main()
