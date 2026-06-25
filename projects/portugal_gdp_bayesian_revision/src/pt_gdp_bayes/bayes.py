"""Small Bayesian utilities used by the Portugal GDP revision model.

The implementation deliberately avoids heavy probabilistic-programming
frameworks. A conjugate Normal-Inverse-Gamma Bayesian linear regression is
sufficient for a transparent first implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import invgamma

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NormalPosterior:
    """Posterior normal distribution parameters."""

    mean: float
    sd: float


@dataclass(frozen=True)
class BayesianLinearRegressionPosterior:
    """Posterior for Bayesian linear regression with NIG prior.

    Model
    -----
    y | beta, sigma² ~ Normal(X beta, sigma² I)
    beta | sigma² ~ Normal(m, sigma² V)
    sigma² ~ InvGamma(a, b)
    """

    beta_mean: FloatArray
    beta_cov_scale: FloatArray
    alpha: float
    beta_scale: float

    def sample_parameters(self, n_samples: int, rng: np.random.Generator | None = None) -> tuple[FloatArray, FloatArray]:
        """Sample regression coefficients and standard deviations.

        Parameters
        ----------
        n_samples:
            Number of posterior draws.
        rng:
            Optional NumPy random generator.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(beta_samples, sigma_samples)``.
        """

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = rng or np.random.default_rng()

        sigma2_samples = invgamma.rvs(
            a=self.alpha,
            scale=self.beta_scale,
            size=n_samples,
            random_state=rng,
        )
        # Vectorised draw: beta_i = mean + sqrt(sigma2_i) * L @ z_i, where L is the
        # Cholesky factor of beta_cov_scale (so cov(L z) = beta_cov_scale, scaled by
        # sigma2_i). Avoids a Python loop over every sample.
        n_features = self.beta_mean.shape[0]
        chol = np.linalg.cholesky(self.beta_cov_scale)
        standard_normals = rng.standard_normal(size=(n_samples, n_features))
        scaled = standard_normals @ chol.T * np.sqrt(sigma2_samples)[:, None]
        beta_samples = self.beta_mean[None, :] + scaled
        return beta_samples, np.sqrt(sigma2_samples)

    def sample_predictive(
        self,
        x_new: ArrayLike,
        n_samples: int,
        *,
        include_observation_noise: bool = True,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """Draw samples from the posterior predictive distribution."""

        rng = rng or np.random.default_rng()
        x = np.asarray(x_new, dtype=float).reshape(-1)
        if x.shape[0] != self.beta_mean.shape[0]:
            raise ValueError(
                f"x_new has {x.shape[0]} elements but model expects {self.beta_mean.shape[0]}"
            )

        beta_samples, sigma_samples = self.sample_parameters(n_samples=n_samples, rng=rng)
        mean_samples = beta_samples @ x
        if include_observation_noise:
            return mean_samples + rng.normal(loc=0.0, scale=sigma_samples, size=n_samples)
        return mean_samples


def _as_float_matrix(x: ArrayLike) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if not np.isfinite(arr).all():
        raise ValueError("X contains non-finite values")
    return arr


def _as_float_vector(y: ArrayLike) -> FloatArray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if not np.isfinite(arr).all():
        raise ValueError("y contains non-finite values")
    return arr


def normal_normal_update(
    *,
    prior_mean: float,
    prior_sd: float,
    observation_mean: float,
    observation_sd: float,
) -> NormalPosterior:
    """Combine a normal prior with a normal observation model.

    Parameters
    ----------
    prior_mean, prior_sd:
        Mean and standard deviation of the prior distribution.
    observation_mean, observation_sd:
        Mean and standard deviation of the observation likelihood.

    Returns
    -------
    NormalPosterior
        Posterior normal distribution.
    """

    if prior_sd <= 0 or observation_sd <= 0:
        raise ValueError("Standard deviations must be positive")

    prior_precision = 1.0 / prior_sd**2
    observation_precision = 1.0 / observation_sd**2
    posterior_variance = 1.0 / (prior_precision + observation_precision)
    posterior_mean = posterior_variance * (
        prior_precision * prior_mean + observation_precision * observation_mean
    )
    return NormalPosterior(mean=float(posterior_mean), sd=float(np.sqrt(posterior_variance)))


def fit_bayesian_linear_regression(
    X: ArrayLike,
    y: ArrayLike,
    *,
    prior_mean: ArrayLike | None = None,
    prior_variance: float = 10.0,
    alpha0: float = 2.0,
    beta0: float = 1.0,
    ridge: float = 1e-8,
) -> BayesianLinearRegressionPosterior:
    """Fit conjugate Bayesian linear regression.

    Parameters
    ----------
    X:
        Design matrix.
    y:
        Response vector.
    prior_mean:
        Optional prior mean for coefficients. Defaults to zeros.
    prior_variance:
        Scalar prior variance for coefficients.
    alpha0, beta0:
        Inverse-gamma prior parameters for residual variance.
    ridge:
        Small numerical ridge added to covariance calculations.

    Returns
    -------
    BayesianLinearRegressionPosterior
        Posterior object with sampling methods.
    """

    X_arr = _as_float_matrix(X)
    y_arr = _as_float_vector(y)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must contain the same number of rows")
    if prior_variance <= 0:
        raise ValueError("prior_variance must be positive")
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError("alpha0 and beta0 must be positive")

    n_obs, n_features = X_arr.shape
    m0 = np.zeros(n_features, dtype=float) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    if m0.shape != (n_features,):
        raise ValueError("prior_mean must have one value per feature")

    V0_inv = np.eye(n_features, dtype=float) / prior_variance
    XtX = X_arr.T @ X_arr
    Vn_inv = V0_inv + XtX + ridge * np.eye(n_features)
    Vn = np.linalg.inv(Vn_inv)
    mn = Vn @ (V0_inv @ m0 + X_arr.T @ y_arr)

    alpha_n = alpha0 + n_obs / 2.0
    quadratic_prior = float(m0.T @ V0_inv @ m0)
    quadratic_post = float(mn.T @ Vn_inv @ mn)
    beta_n = beta0 + 0.5 * (float(y_arr.T @ y_arr) + quadratic_prior - quadratic_post)
    beta_n = max(beta_n, ridge)

    return BayesianLinearRegressionPosterior(
        beta_mean=mn,
        beta_cov_scale=Vn,
        alpha=float(alpha_n),
        beta_scale=float(beta_n),
    )


def summarize_samples(samples: ArrayLike, credible_mass: float = 0.90) -> dict[str, float]:
    """Return mean, median and central credible interval for sample draws."""

    arr = np.asarray(samples, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("samples must not be empty")
    if not 0 < credible_mass < 1:
        raise ValueError("credible_mass must lie between 0 and 1")

    lower_q = (1.0 - credible_mass) / 2.0
    upper_q = 1.0 - lower_q
    lower_pct = lower_q * 100.0
    upper_pct = upper_q * 100.0
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "sd": float(np.std(arr, ddof=1)),
        f"q{lower_pct:g}": float(np.quantile(arr, lower_q)),
        f"q{upper_pct:g}": float(np.quantile(arr, upper_q)),
    }
