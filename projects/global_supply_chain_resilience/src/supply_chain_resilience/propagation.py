"""Fixed-coefficient downstream exposure propagation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import eigs, spsolve


@dataclass(frozen=True)
class AdmissibilityDiagnostics:
    spectral_radius: float
    eigen_residual: float
    admissible: bool


def spectral_radius_diagnostics(coefficients: pd.DataFrame) -> AdmissibilityDiagnostics:
    """Estimate spectral radius and dominant-eigenpair residual for non-negative A."""
    if coefficients.empty or not coefficients.index.equals(coefficients.columns):
        raise ValueError("coefficients must be a non-empty square labelled matrix.")
    values = coefficients.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("coefficients must be finite and non-negative.")

    if len(coefficients) <= 2:
        eigenvalues, eigenvectors = np.linalg.eig(values)
        dominant = int(np.argmax(np.abs(eigenvalues)))
        eigenvalue = eigenvalues[dominant]
        vector = eigenvectors[:, dominant]
        matrix_for_residual = values
    else:
        matrix = csc_matrix(values)
        eigenvalues, eigenvectors = eigs(matrix, k=1, which="LM")
        eigenvalue = eigenvalues[0]
        vector = eigenvectors[:, 0]
        matrix_for_residual = matrix

    spectral_radius = float(abs(eigenvalue))
    numerator = np.linalg.norm(matrix_for_residual @ vector - eigenvalue * vector)
    denominator = max(np.linalg.norm(vector), np.finfo(float).eps)
    eigen_residual = float(numerator / denominator)
    return AdmissibilityDiagnostics(
        spectral_radius=spectral_radius,
        eigen_residual=eigen_residual,
        admissible=spectral_radius < 1.0 - 1e-8 and eigen_residual <= 1e-8,
    )


def solve_downstream_exposure(
    coefficients: pd.DataFrame,
    shock: pd.Series,
    *,
    solver_relative_residual_max: float = 1e-10,
    nonnegative_tolerance: float = -1e-12,
) -> tuple[pd.Series, float]:
    """Solve ``q = s + A.T @ q`` after a separate admissibility gate."""
    if not coefficients.index.equals(coefficients.columns):
        raise ValueError("coefficients must be square with identical labels.")
    if not shock.index.equals(coefficients.index):
        raise ValueError("shock index must match coefficient labels exactly.")
    if not np.isfinite(shock.to_numpy(dtype=float)).all() or (shock < 0.0).any():
        raise ValueError("shock must be finite and non-negative.")

    a_t = csc_matrix(coefficients.to_numpy(dtype=float).T)
    system = eye(len(coefficients), format="csc") - a_t
    rhs = shock.to_numpy(dtype=float)
    solution = np.asarray(spsolve(system, rhs), dtype=float)
    if not np.isfinite(solution).all():
        raise ValueError("propagation solution contains non-finite values.")
    if float(solution.min()) < nonnegative_tolerance:
        raise ValueError("propagation solution violates the non-negativity tolerance.")

    residual = system @ solution - rhs
    denom = max(np.linalg.norm(rhs), np.finfo(float).eps)
    relative_residual = float(np.linalg.norm(residual) / denom)
    if relative_residual > solver_relative_residual_max:
        raise ValueError(
            f"solver relative residual {relative_residual:.3e} exceeds "
            f"{solver_relative_residual_max:.3e}."
        )
    solution = np.where(solution < 0.0, 0.0, solution)
    return pd.Series(solution, index=coefficients.index, dtype=float), relative_residual


def one_node_shock(labels: pd.Index, node: str, fraction: float) -> pd.Series:
    """Build a non-negative one-node proportional shock vector."""
    if node not in labels:
        raise KeyError(node)
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("fraction must be finite and positive.")
    shock = pd.Series(0.0, index=labels, dtype=float)
    shock.loc[node] = float(fraction)
    return shock
