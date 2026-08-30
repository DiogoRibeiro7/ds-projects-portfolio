"""Tests for spatial mobility relocation-cost construction."""

from __future__ import annotations

import numpy as np
import pytest

from mobility_optimization.spatial import normalized_distance_cost_matrix


def test_distance_cost_matrix_is_symmetric_and_zero_diagonal() -> None:
    """Euclidean relocation costs should be symmetric with free self-retention."""
    coordinates = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 10.0]])
    costs = normalized_distance_cost_matrix(coordinates, median_off_diagonal_cost=0.25)

    assert np.allclose(costs, costs.T)
    assert np.allclose(np.diag(costs), 0.0)
    assert np.all(costs[np.triu_indices(3, k=1)] > 0.0)


def test_distance_cost_matrix_preserves_requested_median_cost() -> None:
    """Spatial scaling should preserve the earlier experiment's median price level."""
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    costs = normalized_distance_cost_matrix(coordinates, median_off_diagonal_cost=0.25)
    mask = ~np.eye(3, dtype=bool)

    assert float(np.median(costs[mask])) == pytest.approx(0.25)


def test_invalid_spatial_coordinates_are_rejected() -> None:
    """A relocation matrix needs at least two finite two-dimensional centroids."""
    with pytest.raises(ValueError, match="shape"):
        normalized_distance_cost_matrix([[0.0, 0.0]])
