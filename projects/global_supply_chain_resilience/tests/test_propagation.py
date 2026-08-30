from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supply_chain_resilience.propagation import (
    one_node_shock,
    solve_downstream_exposure,
    spectral_radius_diagnostics,
)


def _coefficients() -> pd.DataFrame:
    labels = ["AAA_A", "BBB_B"]
    return pd.DataFrame(
        [[0.2, 0.1], [0.3, 0.2]],
        index=labels,
        columns=labels,
        dtype=float,
    )


def test_spectral_radius_matches_dense_reference() -> None:
    a = _coefficients()
    expected = float(max(abs(np.linalg.eigvals(a.to_numpy()))))
    diagnostics = spectral_radius_diagnostics(a)

    assert diagnostics.spectral_radius == pytest.approx(expected, rel=1e-10, abs=1e-12)
    assert diagnostics.eigen_residual <= 1e-10
    assert diagnostics.admissible


def test_downstream_solver_matches_dense_linear_system() -> None:
    a = _coefficients()
    shock = one_node_shock(a.index, "AAA_A", 0.1)

    q, residual = solve_downstream_exposure(a, shock)
    expected = np.linalg.solve(np.eye(2) - a.to_numpy().T, shock.to_numpy())

    assert q.to_numpy() == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert residual <= 1e-12
    assert np.allclose(q.to_numpy(), shock.to_numpy() + a.to_numpy().T @ q.to_numpy())


def test_linear_shock_scaling_is_exact() -> None:
    a = _coefficients()
    q5, _ = solve_downstream_exposure(a, one_node_shock(a.index, "AAA_A", 0.05))
    q10, _ = solve_downstream_exposure(a, one_node_shock(a.index, "AAA_A", 0.10))
    q20, _ = solve_downstream_exposure(a, one_node_shock(a.index, "AAA_A", 0.20))

    assert q10.to_numpy() == pytest.approx(2.0 * q5.to_numpy(), rel=1e-12, abs=1e-12)
    assert q20.to_numpy() == pytest.approx(2.0 * q10.to_numpy(), rel=1e-12, abs=1e-12)


def test_one_node_shock_validates_node_and_fraction() -> None:
    labels = pd.Index(["AAA_A", "BBB_B"])
    with pytest.raises(KeyError):
        one_node_shock(labels, "CCC_C", 0.1)
    with pytest.raises(ValueError, match="positive"):
        one_node_shock(labels, "AAA_A", 0.0)
