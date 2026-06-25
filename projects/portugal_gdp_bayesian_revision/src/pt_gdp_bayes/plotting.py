"""Plotting helpers for the Bayesian GDP revision notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plot_posterior_histogram(samples: ArrayLike, title: str, xlabel: str) -> None:
    """Plot a simple posterior histogram with median and 90% interval."""

    arr = np.asarray(samples, dtype=float).reshape(-1)
    lower, median, upper = np.quantile(arr, [0.05, 0.50, 0.95])

    plt.figure(figsize=(9, 5))
    plt.hist(arr, bins=60, alpha=0.75)
    plt.axvline(median, linestyle="--", linewidth=2, label=f"median = {median:,.2f}")
    plt.axvline(lower, linestyle=":", linewidth=2, label=f"5% = {lower:,.2f}")
    plt.axvline(upper, linestyle=":", linewidth=2, label=f"95% = {upper:,.2f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Posterior draws")
    plt.legend()
    plt.tight_layout()
    plt.show()
