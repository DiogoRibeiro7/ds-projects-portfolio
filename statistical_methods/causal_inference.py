"""
Causal inference methods including Instrumental Variables (IV),
Difference-in-Differences (DiD), Regression Discontinuity Design (RDD),
Propensity Score Matching, and Synthetic Control.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class CausalEstimate:
    """
    Container for causal effect estimates returned by estimators.

    Attributes
    ----------
    effect : float
        Estimated causal effect in the same units as the outcome variable.
    standard_error : float
        Standard error of ``effect`` (often computed via asymptotic formulas).
    confidence_interval : Tuple[float, float]
        Two-sided interval (typically 95%) for the effect.
    p_value : float
        Hypothesis test p-value.
    method : str
        Name of the estimator (e.g., ``"2SLS"`` or ``"DiD"``).
    n_treated : int
        Number of treated observations contributing to the estimate.
    n_control : int
        Number of control observations.
    diagnostics : Dict[str, Any] | None
        Optional metadata such as weak-instrument F-statistics.

    Examples
    --------
    >>> CausalEstimate(
    ...     effect=-0.12,
    ...     standard_error=0.03,
    ...     confidence_interval=(-0.18, -0.06),
    ...     p_value=0.001,
    ...     method="2SLS",
    ...     n_treated=500,
    ...     n_control=520,
    ...     diagnostics={"f_statistic": 25.3},
    ... )
    CausalEstimate(effect=-0.12, standard_error=0.03, confidence_interval=(-0.18, -0.06), p_value=0.001, method='2SLS', n_treated=500, n_control=520, diagnostics={'f_statistic': 25.3})
    """

    effect: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    n_treated: int
    n_control: int
    diagnostics: Dict[str, Any] = None


class InstrumentalVariables:
    """Instrumental Variables (IV) estimation methods."""

    def __init__(self):
        """Initialize IV estimator."""
        self.first_stage = None
        self.second_stage = None

    def two_stage_least_squares(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        W: Optional[np.ndarray] = None,
    ) -> CausalEstimate:
        """
        Two-Stage Least Squares (2SLS) estimation.

        Args:
            y: Outcome variable
            X: Endogenous treatment variable
            Z: Instrumental variable(s)
            W: Exogenous covariates (optional)

        Returns:
            CausalEstimate with IV estimates
        """
        n = len(y)

        # Reshape arrays
        y = y.reshape(-1, 1)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z

        # Combine instruments and covariates
        if W is not None:
            W = W.reshape(-1, 1) if W.ndim == 1 else W
            Z_full = np.hstack([Z, W])
        else:
            Z_full = Z

        # First stage: Regress X on Z
        self.first_stage = LinearRegression()
        self.first_stage.fit(Z_full, X)
        X_hat = self.first_stage.predict(Z_full)

        # Test for weak instruments
        f_stat = self._test_weak_instruments(X, Z_full, X_hat)

        # Second stage: Regress y on X_hat
        if W is not None:
            X_hat_full = np.hstack([X_hat, W])
        else:
            X_hat_full = X_hat

        self.second_stage = LinearRegression()
        self.second_stage.fit(X_hat_full, y)
        effect = self.second_stage.coef_[0, 0]

        # Calculate standard errors (simplified - use econometric packages for robust SE)
        residuals = y - self.second_stage.predict(X_hat_full)
        sigma_squared = np.sum(residuals**2) / (n - X_hat_full.shape[1])
        var_matrix = sigma_squared * np.linalg.inv(X_hat_full.T @ X_hat_full)
        se = np.sqrt(var_matrix[0, 0])

        # Calculate statistics
        t_stat = effect / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - X_hat_full.shape[1]))
        ci = (effect - 1.96 * se, effect + 1.96 * se)

        return CausalEstimate(
            effect=float(effect),
            standard_error=float(se),
            confidence_interval=ci,
            p_value=float(p_value),
            method="2SLS",
            n_treated=int(np.sum(X > np.median(X))),
            n_control=int(np.sum(X <= np.median(X))),
            diagnostics={"f_statistic": f_stat, "weak_instrument": f_stat < 10},
        )

    def _test_weak_instruments(
        self, X: np.ndarray, Z: np.ndarray, X_hat: np.ndarray
    ) -> float:
        """Test for weak instruments using F-statistic."""
        n, k = Z.shape
        residuals = X - X_hat
        rss = np.sum(residuals**2)
        tss = np.sum((X - np.mean(X)) ** 2)
        r_squared = 1 - rss / tss

        f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
        return float(f_stat)


class DifferenceInDifferences:
    """Difference-in-Differences (DiD) estimation."""

    def estimate(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        time_col: str,
        group_col: str,
        covariates: Optional[List[str]] = None,
    ) -> CausalEstimate:
        """
        Estimate DiD treatment effect.

        Args:
            data: Panel data DataFrame
            outcome_col: Name of outcome variable
            treatment_col: Name of treatment indicator
            time_col: Name of time period indicator
            group_col: Name of group identifier
            covariates: Optional list of covariate names

        Returns:
            CausalEstimate with DiD estimate
        """
        # Create interaction term
        data = data.copy()
        data["treat_post"] = data[treatment_col] * data[time_col]

        # Build regression formula
        X_cols = [treatment_col, time_col, "treat_post"]
        if covariates:
            X_cols.extend(covariates)

        X = data[X_cols].values
        y = data[outcome_col].values

        # Run regression
        model = LinearRegression()
        model.fit(X, y)

        # The DiD estimate is the coefficient on the interaction term
        did_effect = model.coef_[2]  # treat_post coefficient

        # Calculate standard errors (simplified)
        n = len(y)
        residuals = y - model.predict(X)
        sigma_squared = np.sum(residuals**2) / (n - X.shape[1])
        var_matrix = sigma_squared * np.linalg.inv(X.T @ X)
        se = np.sqrt(var_matrix[2, 2])

        # Calculate statistics
        t_stat = did_effect / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - X.shape[1]))
        ci = (did_effect - 1.96 * se, did_effect + 1.96 * se)

        # Count treated and control
        n_treated = data[data[treatment_col] == 1][group_col].nunique()
        n_control = data[data[treatment_col] == 0][group_col].nunique()

        # Test parallel trends (simplified)
        parallel_trends_test = self._test_parallel_trends(
            data, outcome_col, treatment_col, time_col
        )

        return CausalEstimate(
            effect=float(did_effect),
            standard_error=float(se),
            confidence_interval=ci,
            p_value=float(p_value),
            method="Difference-in-Differences",
            n_treated=n_treated,
            n_control=n_control,
            diagnostics={"parallel_trends_p": parallel_trends_test},
        )

    def _test_parallel_trends(
        self, data: pd.DataFrame, outcome_col: str, treatment_col: str, time_col: str
    ) -> float:
        """Test parallel trends assumption (simplified)."""
        # Check pre-treatment trends
        pre_data = data[data[time_col] == 0]

        if len(pre_data) == 0:
            return np.nan

        treated_pre = pre_data[pre_data[treatment_col] == 1][outcome_col]
        control_pre = pre_data[pre_data[treatment_col] == 0][outcome_col]

        if len(treated_pre) > 0 and len(control_pre) > 0:
            _, p_value = stats.ttest_ind(treated_pre, control_pre)
            return float(p_value)
        return np.nan

    def event_study(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        time_col: str,
        event_time_col: str,
    ) -> pd.DataFrame:
        """
        Perform event study analysis.

        Args:
            data: Panel data
            outcome_col: Outcome variable
            treatment_col: Treatment indicator
            time_col: Time period
            event_time_col: Time relative to treatment

        Returns:
            DataFrame with event study results
        """
        results = []

        for event_time in sorted(data[event_time_col].unique()):
            if event_time == -1:  # Omitted category
                continue

            data_temp = data.copy()
            data_temp[f"event_{event_time}"] = (
                data_temp[event_time_col] == event_time
            ).astype(int)

            X = data_temp[[f"event_{event_time}"]].values
            y = data_temp[outcome_col].values

            model = LinearRegression()
            model.fit(X, y)

            coef = model.coef_[0]
            n = len(y)
            residuals = y - model.predict(X)
            se = np.sqrt(np.sum(residuals**2) / (n - 2) * np.linalg.inv(X.T @ X)[0, 0])

            results.append(
                {
                    "event_time": event_time,
                    "coefficient": coef,
                    "std_error": se,
                    "ci_lower": coef - 1.96 * se,
                    "ci_upper": coef + 1.96 * se,
                }
            )

        return pd.DataFrame(results)


class RegressionDiscontinuity:
    """Regression Discontinuity Design (RDD) estimation."""

    def sharp_rdd(
        self,
        running_var: np.ndarray,
        outcome: np.ndarray,
        cutoff: float,
        bandwidth: Optional[float] = None,
        polynomial: int = 1,
    ) -> CausalEstimate:
        """
        Sharp Regression Discontinuity estimation.

        Args:
            running_var: Running/forcing variable
            outcome: Outcome variable
            cutoff: Cutoff point for treatment
            bandwidth: Bandwidth for local regression
            polynomial: Polynomial order for regression

        Returns:
            CausalEstimate with RDD estimate
        """
        # Center running variable at cutoff
        X = running_var - cutoff
        treatment = (running_var >= cutoff).astype(int)

        # Determine bandwidth if not provided
        if bandwidth is None:
            bandwidth = self._optimal_bandwidth(X, outcome)

        # Select observations within bandwidth
        mask = np.abs(X) <= bandwidth
        X_local = X[mask]
        y_local = outcome[mask]
        treatment_local = treatment[mask]

        # Build polynomial features
        features = [treatment_local]
        for p in range(1, polynomial + 1):
            features.append(X_local**p)
            features.append(treatment_local * (X_local**p))  # Interaction terms

        X_matrix = np.column_stack(features)

        # Estimate model
        model = LinearRegression()
        model.fit(X_matrix, y_local)

        # RDD estimate is coefficient on treatment
        rdd_effect = model.coef_[0]

        # Calculate standard errors
        n = len(y_local)
        residuals = y_local - model.predict(X_matrix)
        sigma_squared = np.sum(residuals**2) / (n - X_matrix.shape[1])
        var_matrix = sigma_squared * np.linalg.inv(X_matrix.T @ X_matrix)
        se = np.sqrt(var_matrix[0, 0])

        # Statistics
        t_stat = rdd_effect / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - X_matrix.shape[1]))
        ci = (rdd_effect - 1.96 * se, rdd_effect + 1.96 * se)

        # Test for manipulation
        manipulation_test = self._test_manipulation(running_var, cutoff)

        return CausalEstimate(
            effect=float(rdd_effect),
            standard_error=float(se),
            confidence_interval=ci,
            p_value=float(p_value),
            method="Sharp RDD",
            n_treated=int(np.sum(treatment_local)),
            n_control=int(np.sum(1 - treatment_local)),
            diagnostics={
                "bandwidth": bandwidth,
                "polynomial": polynomial,
                "manipulation_test_p": manipulation_test,
            },
        )

    def _optimal_bandwidth(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate optimal bandwidth using Imbens-Kalyanaraman method (simplified)."""
        n = len(X)
        X_std = np.std(X)

        # Simplified bandwidth selection
        h = 1.06 * X_std * n ** (-1 / 5)

        return float(h)

    def _test_manipulation(self, running_var: np.ndarray, cutoff: float) -> float:
        """Test for manipulation of running variable at cutoff."""
        # McCrary density test (simplified)
        bins = np.histogram_bin_edges(running_var, bins=30)
        hist_below, _ = np.histogram(running_var[running_var < cutoff], bins=bins)
        hist_above, _ = np.histogram(running_var[running_var >= cutoff], bins=bins)

        # Test for discontinuity in density
        density_below = hist_below[-1] if len(hist_below) > 0 else 0
        density_above = hist_above[0] if len(hist_above) > 0 else 0

        if density_below > 0 and density_above > 0:
            ratio = density_above / density_below
            # Simplified test statistic
            z_stat = (ratio - 1) / 0.5  # Simplified SE
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            return float(p_value)
        return np.nan

    def fuzzy_rdd(
        self,
        running_var: np.ndarray,
        outcome: np.ndarray,
        treatment: np.ndarray,
        cutoff: float,
        bandwidth: Optional[float] = None,
    ) -> CausalEstimate:
        """
        Fuzzy Regression Discontinuity estimation.

        Args:
            running_var: Running variable
            outcome: Outcome variable
            treatment: Actual treatment status
            cutoff: Cutoff point
            bandwidth: Bandwidth for local regression

        Returns:
            CausalEstimate with fuzzy RDD estimate
        """
        # This is essentially IV with cutoff as instrument
        Z = (running_var >= cutoff).astype(int)

        iv_estimator = InstrumentalVariables()
        return iv_estimator.two_stage_least_squares(outcome, treatment, Z)


class PropensityScoreMatching:
    """Propensity Score Matching methods."""

    def __init__(self):
        """Initialize PSM estimator."""
        self.propensity_model = None
        self.matches = None

    def estimate_ate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        n_neighbors: int = 1,
        caliper: Optional[float] = None,
    ) -> CausalEstimate:
        """
        Estimate Average Treatment Effect using PSM.

        Args:
            X: Covariates
            treatment: Treatment indicator
            outcome: Outcome variable
            n_neighbors: Number of neighbors for matching
            caliper: Maximum distance for matching

        Returns:
            CausalEstimate with PSM estimate
        """
        # Estimate propensity scores
        self.propensity_model = LogisticRegression(max_iter=1000)
        self.propensity_model.fit(X, treatment)
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]

        # Check overlap
        overlap_test = self._check_overlap(propensity_scores, treatment)

        # Perform matching
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        # Match each treated unit to control units
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(propensity_scores[control_idx].reshape(-1, 1))

        distances, indices = nn.kneighbors(
            propensity_scores[treated_idx].reshape(-1, 1)
        )

        # Apply caliper if specified
        if caliper is not None:
            valid_matches = distances[:, 0] <= caliper
        else:
            valid_matches = np.ones(len(treated_idx), dtype=bool)

        # Calculate treatment effect
        treatment_effects = []
        for i, treated in enumerate(treated_idx):
            if valid_matches[i]:
                control_matches = control_idx[indices[i]]
                effect = outcome[treated] - np.mean(outcome[control_matches])
                treatment_effects.append(effect)

        if not treatment_effects:
            return CausalEstimate(
                effect=np.nan,
                standard_error=np.nan,
                confidence_interval=(np.nan, np.nan),
                p_value=np.nan,
                method="Propensity Score Matching",
                n_treated=len(treated_idx),
                n_control=len(control_idx),
                diagnostics={"error": "No valid matches found"},
            )

        ate = np.mean(treatment_effects)
        se = np.std(treatment_effects) / np.sqrt(len(treatment_effects))

        # Calculate statistics
        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(treatment_effects) - 1))
        ci = (ate - 1.96 * se, ate + 1.96 * se)

        return CausalEstimate(
            effect=float(ate),
            standard_error=float(se),
            confidence_interval=ci,
            p_value=float(p_value),
            method="Propensity Score Matching",
            n_treated=int(np.sum(valid_matches)),
            n_control=len(control_idx),
            diagnostics={
                "overlap": overlap_test,
                "n_matched": int(np.sum(valid_matches)),
                "caliper_used": caliper is not None,
            },
        )

    def _check_overlap(
        self, propensity_scores: np.ndarray, treatment: np.ndarray
    ) -> float:
        """Check overlap in propensity scores."""
        treated_ps = propensity_scores[treatment == 1]
        control_ps = propensity_scores[treatment == 0]

        # Calculate overlap as proportion of treated within control range
        control_min, control_max = control_ps.min(), control_ps.max()
        overlap = np.mean((treated_ps >= control_min) & (treated_ps <= control_max))

        return float(overlap)

    def inverse_propensity_weighting(
        self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
    ) -> CausalEstimate:
        """
        Inverse Propensity Score Weighting (IPW) estimation.

        Args:
            X: Covariates
            treatment: Treatment indicator
            outcome: Outcome variable

        Returns:
            CausalEstimate with IPW estimate
        """
        # Estimate propensity scores
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X, treatment)
        ps = ps_model.predict_proba(X)[:, 1]

        # Clip propensity scores to avoid extreme weights
        ps = np.clip(ps, 0.01, 0.99)

        # Calculate IPW weights
        weights = treatment / ps + (1 - treatment) / (1 - ps)

        # Weighted outcomes
        weighted_treated = np.sum(outcome * treatment / ps) / np.sum(treatment / ps)
        weighted_control = np.sum(outcome * (1 - treatment) / (1 - ps)) / np.sum(
            (1 - treatment) / (1 - ps)
        )

        ate = weighted_treated - weighted_control

        # Bootstrap standard errors
        n_bootstrap = 1000
        bootstrap_ates = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            treatment_boot = treatment[idx]
            outcome_boot = outcome[idx]

            ps_model_boot = LogisticRegression(max_iter=1000)
            ps_model_boot.fit(X_boot, treatment_boot)
            ps_boot = ps_model_boot.predict_proba(X_boot)[:, 1]
            ps_boot = np.clip(ps_boot, 0.01, 0.99)

            weighted_treated_boot = np.sum(
                outcome_boot * treatment_boot / ps_boot
            ) / np.sum(treatment_boot / ps_boot)
            weighted_control_boot = np.sum(
                outcome_boot * (1 - treatment_boot) / (1 - ps_boot)
            ) / np.sum((1 - treatment_boot) / (1 - ps_boot))

            bootstrap_ates.append(weighted_treated_boot - weighted_control_boot)

        se = np.std(bootstrap_ates)
        ci = tuple(np.percentile(bootstrap_ates, [2.5, 97.5]))

        # Calculate p-value
        z_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return CausalEstimate(
            effect=float(ate),
            standard_error=float(se),
            confidence_interval=ci,
            p_value=float(p_value),
            method="Inverse Propensity Weighting",
            n_treated=int(np.sum(treatment)),
            n_control=int(np.sum(1 - treatment)),
            diagnostics={"max_weight": float(np.max(weights))},
        )


class SyntheticControl:
    """Synthetic Control Method for causal inference."""

    def __init__(self):
        """Initialize Synthetic Control estimator."""
        self.weights = None

    def estimate(
        self, treated_unit: np.ndarray, control_units: np.ndarray, pre_period: int
    ) -> CausalEstimate:
        """
        Estimate treatment effect using Synthetic Control.

        Args:
            treated_unit: Time series for treated unit
            control_units: Matrix of control unit time series (units x time)
            pre_period: Number of pre-treatment periods

        Returns:
            CausalEstimate with synthetic control estimate
        """
        # Optimize weights to minimize pre-treatment fit
        self.weights = self._optimize_weights(
            treated_unit[:pre_period], control_units[:, :pre_period]
        )

        # Create synthetic control
        synthetic_control = np.sum(control_units * self.weights[:, np.newaxis], axis=0)

        # Calculate treatment effect
        post_period = len(treated_unit) - pre_period
        treatment_effect = np.mean(
            treated_unit[pre_period:] - synthetic_control[pre_period:]
        )

        # Placebo tests for inference
        placebo_effects = self._placebo_tests(control_units, pre_period)
        p_value = np.mean(np.abs(placebo_effects) >= np.abs(treatment_effect))

        # Standard error from placebo distribution
        se = np.std(placebo_effects)
        ci = (treatment_effect - 1.96 * se, treatment_effect + 1.96 * se)

        # Calculate pre-treatment fit
        pre_rmse = np.sqrt(
            np.mean((treated_unit[:pre_period] - synthetic_control[:pre_period]) ** 2)
        )

        return CausalEstimate(
            effect=float(treatment_effect),
            standard_error=float(se),
            confidence_interval=ci,
            p_value=float(p_value),
            method="Synthetic Control",
            n_treated=1,
            n_control=len(control_units),
            diagnostics={
                "weights": self.weights,
                "pre_treatment_rmse": float(pre_rmse),
                "n_control_units": len(control_units),
            },
        )

    def _optimize_weights(
        self, treated_pre: np.ndarray, control_pre: np.ndarray
    ) -> np.ndarray:
        """Optimize weights for synthetic control."""
        n_controls = control_pre.shape[0]

        # Objective function: minimize squared differences
        def objective(w):
            synthetic = np.sum(control_pre * w[:, np.newaxis], axis=0)
            return np.sum((treated_pre - synthetic) ** 2)

        # Constraints: weights sum to 1 and are non-negative
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_controls)]

        # Initial weights
        initial_weights = np.ones(n_controls) / n_controls

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def _placebo_tests(self, control_units: np.ndarray, pre_period: int) -> np.ndarray:
        """Run placebo tests on control units."""
        placebo_effects = []

        for i in range(len(control_units)):
            # Treat control unit i as treated
            placebo_treated = control_units[i]
            placebo_controls = np.delete(control_units, i, axis=0)

            # Estimate synthetic control
            weights = self._optimize_weights(
                placebo_treated[:pre_period], placebo_controls[:, :pre_period]
            )

            synthetic = np.sum(placebo_controls * weights[:, np.newaxis], axis=0)
            effect = np.mean(placebo_treated[pre_period:] - synthetic[pre_period:])
            placebo_effects.append(effect)

        return np.array(placebo_effects)


if __name__ == "__main__":
    # Example usage
    print("Causal Inference Methods")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n = 1000

    # Instrumental Variables example
    print("\nInstrumental Variables (2SLS):")
    Z = np.random.normal(0, 1, n)  # Instrument
    X = 0.5 * Z + np.random.normal(0, 0.5, n)  # Endogenous treatment
    y = 2 * X + np.random.normal(0, 1, n)  # Outcome

    iv = InstrumentalVariables()
    iv_result = iv.two_stage_least_squares(y, X, Z)
    print(f"  Effect: {iv_result.effect:.3f}")
    print(f"  SE: {iv_result.standard_error:.3f}")
    print(f"  P-value: {iv_result.p_value:.4f}")
    print(f"  Weak instrument: {iv_result.diagnostics['weak_instrument']}")

    # Difference-in-Differences example
    print("\nDifference-in-Differences:")
    did_data = pd.DataFrame(
        {
            "outcome": np.random.normal(100, 15, 400),
            "treatment": np.repeat([0, 0, 1, 1], 100),
            "time": np.tile([0, 1, 0, 1], 100),
            "group": np.repeat(range(200), 2),
        }
    )
    # Add treatment effect
    did_data.loc[(did_data["treatment"] == 1) & (did_data["time"] == 1), "outcome"] += (
        10
    )

    did = DifferenceInDifferences()
    did_result = did.estimate(did_data, "outcome", "treatment", "time", "group")
    print(f"  Effect: {did_result.effect:.3f}")
    print(f"  SE: {did_result.standard_error:.3f}")
    print(f"  P-value: {did_result.p_value:.4f}")

    # Regression Discontinuity example
    print("\nRegression Discontinuity Design:")
    running_var = np.random.uniform(-2, 2, n)
    treatment_effect = 5
    outcome = (
        3
        + 2 * running_var
        + treatment_effect * (running_var >= 0)
        + np.random.normal(0, 1, n)
    )

    rdd = RegressionDiscontinuity()
    rdd_result = rdd.sharp_rdd(running_var, outcome, cutoff=0)
    print(f"  Effect: {rdd_result.effect:.3f}")
    print(f"  SE: {rdd_result.standard_error:.3f}")
    print(f"  P-value: {rdd_result.p_value:.4f}")

    # Propensity Score Matching example
    print("\nPropensity Score Matching:")
    X_cov = np.random.normal(0, 1, (n, 5))
    treatment = (X_cov[:, 0] + np.random.normal(0, 1, n) > 0).astype(int)
    outcome_psm = 3 + 2 * treatment + X_cov[:, 0] + np.random.normal(0, 1, n)

    psm = PropensityScoreMatching()
    psm_result = psm.estimate_ate(X_cov, treatment, outcome_psm)
    print(f"  Effect: {psm_result.effect:.3f}")
    print(f"  SE: {psm_result.standard_error:.3f}")
    print(f"  P-value: {psm_result.p_value:.4f}")
