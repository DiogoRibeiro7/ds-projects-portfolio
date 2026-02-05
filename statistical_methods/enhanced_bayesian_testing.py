"""Enhanced Bayesian A/B testing with network effects, time-dependent methods,
and advanced capabilities.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

warnings.filterwarnings("ignore")


@dataclass
class NetworkEffectResult:
    """Results from network effect corrected testing."""

    direct_effect: float
    indirect_effect: float
    total_effect: float
    spillover_ratio: float
    confidence_intervals: dict[str, tuple[float, float]]
    network_statistics: dict[str, Any]
    p_values: dict[str, float]


@dataclass
class TimeDependentResult:
    """Results from time-dependent Bayesian testing."""

    time_series_effects: pd.DataFrame
    cumulative_effect: float
    peak_effect: tuple[float, float]  # (time, effect)
    decay_rate: float | None
    seasonality_detected: bool
    trend_component: float | None
    confidence_bands: pd.DataFrame


class NetworkEffectBayesianTest:
    """Bayesian A/B testing with network effect corrections."""

    def __init__(self, prior_alpha: float = 1, prior_beta: float = 1):
        """Initialize network effect Bayesian testing.

        Args:
            prior_alpha: Alpha parameter for Beta prior
            prior_beta: Beta parameter for Beta prior
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def test_with_network_effects(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        user_col: str,
        network_adjacency: np.ndarray | None = None,
        network_edges: list[tuple[Any, Any]] | None = None,
        n_samples: int = 50000,
    ) -> NetworkEffectResult:
        """Bayesian A/B test accounting for network effects.

        Args:
            data: DataFrame with experiment data
            outcome_col: Outcome column name
            treatment_col: Treatment assignment column
            user_col: User identifier column
            network_adjacency: Adjacency matrix of network
            network_edges: List of edges (user1, user2) if no adjacency matrix
            n_samples: Number of posterior samples

        Returns:
            NetworkEffectResult with direct and indirect effects
        """
        # Build network structure
        if network_adjacency is None and network_edges is not None:
            network_adjacency = self._build_adjacency_matrix(
                data[user_col].unique(), network_edges
            )
        elif network_adjacency is None:
            raise ValueError(
                "Either network_adjacency or network_edges must be provided"
            )

        # Calculate exposure to treatment through network
        user_to_idx = {user: i for i, user in enumerate(data[user_col].unique())}
        treatment_vector = np.zeros(len(user_to_idx))

        for _, row in data.iterrows():
            if row[treatment_col] == 1:
                treatment_vector[user_to_idx[row[user_col]]] = 1

        # Calculate network exposure (proportion of treated neighbors)
        network_exposure = network_adjacency @ treatment_vector
        degree = network_adjacency.sum(axis=1)
        network_exposure = np.divide(
            network_exposure,
            degree,
            out=np.zeros_like(network_exposure),
            where=degree != 0,
        )

        # Add network exposure to data
        data = data.copy()
        data["network_exposure"] = data[user_col].map(
            lambda u: network_exposure[user_to_idx[u]]
        )

        # Stratify by exposure levels
        exposure_levels = [0, 0.25, 0.5, 0.75, 1.0]
        data["exposure_bin"] = pd.cut(
            data["network_exposure"],
            bins=[-0.01] + exposure_levels,
            labels=["0%", "0-25%", "25-50%", "50-75%", "75-100%"],
        )

        # Estimate effects at different exposure levels
        direct_effects = []
        indirect_effects = []

        for exposure_bin in data["exposure_bin"].unique():
            subset = data[data["exposure_bin"] == exposure_bin]

            if len(subset) == 0:
                continue

            # Direct effect: treatment vs control at this exposure level
            treated = subset[subset[treatment_col] == 1]
            control = subset[subset[treatment_col] == 0]

            if len(treated) > 0 and len(control) > 0:
                # Bayesian estimation
                conv_t = treated[outcome_col].sum()
                n_t = len(treated)
                conv_c = control[outcome_col].sum()
                n_c = len(control)

                # Sample from posteriors
                samples_t = np.random.beta(
                    self.prior_alpha + conv_t, self.prior_beta + n_t - conv_t, n_samples
                )
                samples_c = np.random.beta(
                    self.prior_alpha + conv_c, self.prior_beta + n_c - conv_c, n_samples
                )

                direct_effect = np.mean(samples_t - samples_c)
                direct_effects.append(direct_effect)

                # Indirect effect: effect of network exposure within control group
                if exposure_bin != "0%":
                    no_exposure = data[
                        (data["exposure_bin"] == "0%") & (data[treatment_col] == 0)
                    ]
                    if len(no_exposure) > 0:
                        conv_no_exp = no_exposure[outcome_col].sum()
                        n_no_exp = len(no_exposure)

                        samples_no_exp = np.random.beta(
                            self.prior_alpha + conv_no_exp,
                            self.prior_beta + n_no_exp - conv_no_exp,
                            n_samples,
                        )

                        indirect_effect = np.mean(samples_c - samples_no_exp)
                        indirect_effects.append(indirect_effect)

        # Aggregate effects
        avg_direct_effect = np.mean(direct_effects) if direct_effects else 0
        avg_indirect_effect = np.mean(indirect_effects) if indirect_effects else 0
        total_effect = avg_direct_effect + avg_indirect_effect

        # Calculate spillover ratio
        spillover_ratio = avg_indirect_effect / total_effect if total_effect != 0 else 0

        # Bootstrap confidence intervals
        ci_direct = self._bootstrap_ci(direct_effects) if direct_effects else (0, 0)
        ci_indirect = (
            self._bootstrap_ci(indirect_effects) if indirect_effects else (0, 0)
        )
        ci_total = (ci_direct[0] + ci_indirect[0], ci_direct[1] + ci_indirect[1])

        # Network statistics
        avg_degree = degree.mean()
        clustering_coeff = self._calculate_clustering_coefficient(network_adjacency)

        # Calculate p-values using permutation test
        p_value_direct = self._permutation_test(
            data, outcome_col, treatment_col, avg_direct_effect
        )
        p_value_indirect = self._permutation_test_network(
            data, outcome_col, "network_exposure", avg_indirect_effect
        )

        return NetworkEffectResult(
            direct_effect=avg_direct_effect,
            indirect_effect=avg_indirect_effect,
            total_effect=total_effect,
            spillover_ratio=spillover_ratio,
            confidence_intervals={
                "direct": ci_direct,
                "indirect": ci_indirect,
                "total": ci_total,
            },
            network_statistics={
                "avg_degree": avg_degree,
                "clustering_coefficient": clustering_coeff,
                "n_nodes": len(user_to_idx),
                "n_edges": int(network_adjacency.sum() / 2),
            },
            p_values={"direct": p_value_direct, "indirect": p_value_indirect},
        )

    def _build_adjacency_matrix(
        self, users: np.ndarray, edges: list[tuple[Any, Any]]
    ) -> np.ndarray:
        """Build adjacency matrix from edge list."""
        user_to_idx = {user: i for i, user in enumerate(users)}
        n = len(users)
        adj_matrix = np.zeros((n, n))

        for u1, u2 in edges:
            if u1 in user_to_idx and u2 in user_to_idx:
                i, j = user_to_idx[u1], user_to_idx[u2]
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Symmetric for undirected graph

        return adj_matrix

    def _calculate_clustering_coefficient(self, adj_matrix: np.ndarray) -> float:
        """Calculate global clustering coefficient."""
        n = adj_matrix.shape[0]
        triangles = np.trace(adj_matrix @ adj_matrix @ adj_matrix) / 6

        # Count connected triples
        degree = adj_matrix.sum(axis=1)
        triples = np.sum(degree * (degree - 1)) / 2

        if triples > 0:
            return 3 * triangles / triples
        return 0

    def _bootstrap_ci(
        self, values: list[float], confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if not values:
            return (0, 0)

        n_bootstrap = 10000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        return tuple(
            np.percentile(bootstrap_means, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        )

    def _permutation_test(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        observed_effect: float,
        n_permutations: int = 1000,
    ) -> float:
        """Permutation test for significance."""
        permuted_effects = []

        for _ in range(n_permutations):
            data_perm = data.copy()
            data_perm[treatment_col] = np.random.permutation(data_perm[treatment_col])

            treated = data_perm[data_perm[treatment_col] == 1][outcome_col].mean()
            control = data_perm[data_perm[treatment_col] == 0][outcome_col].mean()
            permuted_effects.append(treated - control)

        p_value = np.mean(np.abs(permuted_effects) >= np.abs(observed_effect))
        return float(p_value)

    def _permutation_test_network(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        exposure_col: str,
        observed_effect: float,
        n_permutations: int = 1000,
    ) -> float:
        """Permutation test for network effects."""
        permuted_effects = []

        for _ in range(n_permutations):
            data_perm = data.copy()
            data_perm[exposure_col] = np.random.permutation(data_perm[exposure_col])

            high_exposure = data_perm[data_perm[exposure_col] > 0.5][outcome_col].mean()
            low_exposure = data_perm[data_perm[exposure_col] <= 0.5][outcome_col].mean()
            permuted_effects.append(high_exposure - low_exposure)

        p_value = np.mean(np.abs(permuted_effects) >= np.abs(observed_effect))
        return float(p_value)


class TimeDependentBayesianTest:
    """Time-dependent Bayesian testing with temporal patterns."""

    def __init__(self, prior_alpha: float = 1, prior_beta: float = 1):
        """Initialize time-dependent Bayesian testing."""
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.gp_model = None

    def test_with_temporal_effects(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        time_col: str,
        user_col: str | None = None,
        granularity: str = "daily",
        n_samples: int = 10000,
    ) -> TimeDependentResult:
        """Perform time-dependent Bayesian testing.

        Args:
            data: DataFrame with temporal experiment data
            outcome_col: Outcome column name
            treatment_col: Treatment assignment column
            time_col: Time column (datetime)
            user_col: Optional user identifier for cohort analysis
            granularity: Time granularity ('hourly', 'daily', 'weekly')
            n_samples: Number of posterior samples

        Returns:
            TimeDependentResult with temporal patterns
        """
        # Convert time column to datetime if needed
        data = data.copy()
        data[time_col] = pd.to_datetime(data[time_col])

        # Aggregate by time period
        if granularity == "hourly":
            data["time_period"] = data[time_col].dt.floor("H")
        elif granularity == "daily":
            data["time_period"] = data[time_col].dt.date
        elif granularity == "weekly":
            data["time_period"] = data[time_col].dt.to_period("W").dt.start_time
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        # Calculate effects over time
        time_effects = []
        time_points = sorted(data["time_period"].unique())

        for t in time_points:
            period_data = data[data["time_period"] == t]

            treated = period_data[period_data[treatment_col] == 1]
            control = period_data[period_data[treatment_col] == 0]

            if len(treated) > 0 and len(control) > 0:
                # Binary outcome
                if data[outcome_col].dtype == bool or set(
                    data[outcome_col].unique()
                ) == {0, 1}:
                    conv_t = treated[outcome_col].sum()
                    n_t = len(treated)
                    conv_c = control[outcome_col].sum()
                    n_c = len(control)

                    # Posterior samples
                    samples_t = np.random.beta(
                        self.prior_alpha + conv_t,
                        self.prior_beta + n_t - conv_t,
                        n_samples,
                    )
                    samples_c = np.random.beta(
                        self.prior_alpha + conv_c,
                        self.prior_beta + n_c - conv_c,
                        n_samples,
                    )
                else:
                    # Continuous outcome - use normal approximation
                    mean_t = treated[outcome_col].mean()
                    std_t = treated[outcome_col].std()
                    mean_c = control[outcome_col].mean()
                    std_c = control[outcome_col].std()

                    samples_t = np.random.normal(
                        mean_t, std_t / np.sqrt(len(treated)), n_samples
                    )
                    samples_c = np.random.normal(
                        mean_c, std_c / np.sqrt(len(control)), n_samples
                    )

                effect = np.mean(samples_t - samples_c)
                ci_lower = np.percentile(samples_t - samples_c, 2.5)
                ci_upper = np.percentile(samples_t - samples_c, 97.5)

                time_effects.append(
                    {
                        "time": t,
                        "effect": effect,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "n_treated": len(treated),
                        "n_control": len(control),
                    }
                )

        # Convert to DataFrame
        effects_df = pd.DataFrame(time_effects)

        # Fit Gaussian Process for smooth effect estimation
        if len(effects_df) > 3:
            X_time = np.arange(len(effects_df)).reshape(-1, 1)
            y_effect = effects_df["effect"].values

            # Configure GP kernel
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
            ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

            self.gp_model = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=10, alpha=1e-6
            )
            self.gp_model.fit(X_time, y_effect)

            # Predict smooth effects with confidence bands
            X_pred = np.linspace(0, len(effects_df) - 1, 100).reshape(-1, 1)
            y_pred, y_std = self.gp_model.predict(X_pred, return_std=True)

            confidence_bands = pd.DataFrame(
                {
                    "time_index": X_pred.flatten(),
                    "smooth_effect": y_pred,
                    "ci_lower": y_pred - 1.96 * y_std,
                    "ci_upper": y_pred + 1.96 * y_std,
                }
            )
        else:
            confidence_bands = pd.DataFrame()

        # Analyze temporal patterns
        cumulative_effect = effects_df["effect"].mean()

        # Find peak effect
        peak_idx = effects_df["effect"].idxmax()
        peak_effect = (
            effects_df.loc[peak_idx, "time"],
            effects_df.loc[peak_idx, "effect"],
        )

        # Detect decay
        decay_rate = self._estimate_decay_rate(effects_df)

        # Detect seasonality
        seasonality = self._detect_seasonality(effects_df)

        # Estimate trend
        trend = self._estimate_trend(effects_df)

        return TimeDependentResult(
            time_series_effects=effects_df,
            cumulative_effect=cumulative_effect,
            peak_effect=peak_effect,
            decay_rate=decay_rate,
            seasonality_detected=seasonality,
            trend_component=trend,
            confidence_bands=confidence_bands,
        )

    def _estimate_decay_rate(self, effects_df: pd.DataFrame) -> float | None:
        """Estimate exponential decay rate if present."""
        if len(effects_df) < 5:
            return None

        # Check if effects decrease over time
        correlation = np.corrcoef(
            np.arange(len(effects_df)), effects_df["effect"].values
        )[0, 1]

        if correlation < -0.5:  # Negative correlation suggests decay
            # Fit exponential decay: effect = a * exp(-b * t)
            def exp_decay(t, a, b):
                return a * np.exp(-b * t)

            from scipy.optimize import curve_fit

            try:
                t = np.arange(len(effects_df))
                y = effects_df["effect"].values

                # Initial guess
                p0 = [y[0], 0.1]

                popt, _ = curve_fit(exp_decay, t, y, p0=p0, maxfev=5000)

                return float(popt[1])  # Return decay rate
            except:
                return None

        return None

    def _detect_seasonality(self, effects_df: pd.DataFrame) -> bool:
        """Detect if there's seasonality in the effects."""
        if len(effects_df) < 7:
            return False

        # Use autocorrelation to detect seasonality
        from statsmodels.tsa.stattools import acf

        effects = effects_df["effect"].values
        autocorr = acf(effects, nlags=min(len(effects) // 2, 20))

        # Look for significant peaks in autocorrelation
        significant_peaks = np.where(np.abs(autocorr[1:]) > 2 / np.sqrt(len(effects)))[
            0
        ]

        # If we have regular peaks, there might be seasonality
        if len(significant_peaks) > 1:
            peak_diffs = np.diff(significant_peaks)
            if np.std(peak_diffs) / np.mean(peak_diffs) < 0.3:  # Regular spacing
                return True

        return False

    def _estimate_trend(self, effects_df: pd.DataFrame) -> float | None:
        """Estimate linear trend in effects."""
        if len(effects_df) < 3:
            return None

        # Simple linear regression
        X = np.arange(len(effects_df)).reshape(-1, 1)
        y = effects_df["effect"].values

        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X, y)

        return float(model.coef_[0])

    def plot_temporal_results(self, result: TimeDependentResult):
        """Plot temporal testing results.

        Args:
            result: TimeDependentResult to visualize
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Time series of effects
        ax = axes[0, 0]
        effects = result.time_series_effects
        ax.plot(range(len(effects)), effects["effect"], "o-", label="Observed")
        ax.fill_between(
            range(len(effects)),
            effects["ci_lower"],
            effects["ci_upper"],
            alpha=0.3,
            label="95% CI",
        )

        # Add smooth GP fit if available
        if not result.confidence_bands.empty:
            bands = result.confidence_bands
            ax.plot(
                bands["time_index"],
                bands["smooth_effect"],
                "r-",
                label="GP Smooth",
                alpha=0.7,
            )
            ax.fill_between(
                bands["time_index"],
                bands["ci_lower"],
                bands["ci_upper"],
                alpha=0.2,
                color="red",
            )

        ax.set_xlabel("Time Period")
        ax.set_ylabel("Treatment Effect")
        ax.set_title("Treatment Effect Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cumulative effect
        ax = axes[0, 1]
        cumul_effects = effects["effect"].cumsum() / np.arange(1, len(effects) + 1)
        ax.plot(range(len(effects)), cumul_effects, "g-", linewidth=2)
        ax.axhline(
            result.cumulative_effect,
            color="red",
            linestyle="--",
            label=f"Mean: {result.cumulative_effect:.3f}",
        )
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Cumulative Average Effect")
        ax.set_title("Cumulative Treatment Effect")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sample sizes over time
        ax = axes[1, 0]
        ax.bar(range(len(effects)), effects["n_treated"], alpha=0.7, label="Treated")
        ax.bar(
            range(len(effects)),
            effects["n_control"],
            alpha=0.7,
            label="Control",
            bottom=effects["n_treated"],
        )
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Sample Size")
        ax.set_title("Sample Sizes Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Summary statistics
        ax = axes[1, 1]
        ax.axis("off")

        summary_text = f"""
        Temporal Analysis Summary:

        Cumulative Effect: {result.cumulative_effect:.4f}
        Peak Effect: {result.peak_effect[1]:.4f} at period {result.peak_effect[0]}
        Decay Rate: {result.decay_rate:.4f if result.decay_rate else 'Not detected'}
        Seasonality: {"Detected" if result.seasonality_detected else "Not detected"}
        Trend: {result.trend_component:.4f if result.trend_component else 'None'}
        Total Periods: {len(effects)}
        """

        ax.text(0.1, 0.5, summary_text, fontsize=11, family="monospace")

        plt.tight_layout()
        plt.show()


class MetaAnalysisBayesian:
    """Meta-analysis tools for combining multiple Bayesian experiments."""

    def __init__(self):
        """Initialize meta-analysis tools."""
        pass

    def combine_experiments(
        self,
        experiments: list[dict[str, Any]],
        method: str = "fixed_effects",
        tau_prior: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Combine multiple Bayesian experiments using meta-analysis.

        Args:
            experiments: List of experiment results, each with:
                - 'effect': estimated effect
                - 'variance': variance of estimate
                - 'n': sample size
                - 'name': experiment name
            method: 'fixed_effects' or 'random_effects'
            tau_prior: Prior for between-study variance (random effects)

        Returns:
            Combined meta-analysis results
        """
        effects = np.array([exp["effect"] for exp in experiments])
        variances = np.array([exp["variance"] for exp in experiments])
        weights = 1 / variances

        if method == "fixed_effects":
            # Fixed effects meta-analysis
            pooled_effect = np.sum(weights * effects) / np.sum(weights)
            pooled_variance = 1 / np.sum(weights)
            pooled_se = np.sqrt(pooled_variance)

            # Test for heterogeneity (Cochran's Q)
            q_stat = np.sum(weights * (effects - pooled_effect) ** 2)
            df = len(effects) - 1
            p_heterogeneity = 1 - stats.chi2.cdf(q_stat, df)

            # I-squared statistic
            i_squared = max(0, (q_stat - df) / q_stat) if q_stat > 0 else 0

        elif method == "random_effects":
            # Random effects meta-analysis (DerSimonian-Laird)

            # Estimate tau-squared (between-study variance)
            q_stat = np.sum(
                weights * (effects - np.sum(weights * effects) / np.sum(weights)) ** 2
            )
            c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
            tau_squared = max(0, (q_stat - (len(effects) - 1)) / c)

            # Adjust weights for random effects
            weights_re = 1 / (variances + tau_squared)
            pooled_effect = np.sum(weights_re * effects) / np.sum(weights_re)
            pooled_variance = 1 / np.sum(weights_re)
            pooled_se = np.sqrt(pooled_variance)

            # Heterogeneity statistics
            df = len(effects) - 1
            p_heterogeneity = 1 - stats.chi2.cdf(q_stat, df)
            i_squared = max(0, (q_stat - df) / q_stat) if q_stat > 0 else 0

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate confidence interval and p-value
        z_score = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se

        # Forest plot data
        forest_data = []
        for exp in experiments:
            exp_ci_lower = exp["effect"] - 1.96 * np.sqrt(exp["variance"])
            exp_ci_upper = exp["effect"] + 1.96 * np.sqrt(exp["variance"])
            forest_data.append(
                {
                    "name": exp["name"],
                    "effect": exp["effect"],
                    "ci_lower": exp_ci_lower,
                    "ci_upper": exp_ci_upper,
                    "weight": 1 / exp["variance"]
                    if method == "fixed_effects"
                    else 1 / (exp["variance"] + tau_squared),
                }
            )

        return {
            "pooled_effect": pooled_effect,
            "pooled_se": pooled_se,
            "confidence_interval": (ci_lower, ci_upper),
            "p_value": p_value,
            "heterogeneity": {
                "q_statistic": q_stat,
                "p_value": p_heterogeneity,
                "i_squared": i_squared,
                "tau_squared": tau_squared if method == "random_effects" else None,
            },
            "forest_data": forest_data,
            "method": method,
            "n_experiments": len(experiments),
            "total_n": sum(exp.get("n", 0) for exp in experiments),
        }

    def create_forest_plot(self, meta_result: dict[str, Any]):
        """Create forest plot for meta-analysis results.

        Args:
            meta_result: Result from combine_experiments
        """
        forest_data = meta_result["forest_data"]
        n = len(forest_data)

        fig, ax = plt.subplots(figsize=(12, max(6, n * 0.5)))

        # Plot individual studies
        y_pos = np.arange(n)

        for i, study in enumerate(forest_data):
            # Confidence interval
            ax.plot([study["ci_lower"], study["ci_upper"]], [i, i], "k-", linewidth=1)

            # Point estimate (size proportional to weight)
            size = study["weight"] / max(s["weight"] for s in forest_data) * 200
            ax.scatter(study["effect"], i, s=size, color="blue", zorder=3)

            # Study name
            ax.text(ax.get_xlim()[0], i, study["name"], ha="right", va="center")

        # Add pooled estimate
        pooled = meta_result["pooled_effect"]
        ci = meta_result["confidence_interval"]

        ax.axvspan(ci[0], ci[1], alpha=0.2, color="red", ymin=-0.5, ymax=n - 0.5)
        ax.axvline(
            pooled, color="red", linestyle="--", linewidth=2, label="Pooled estimate"
        )

        # Add diamond for pooled estimate
        diamond_y = n
        diamond = plt.Polygon(
            [
                [ci[0], diamond_y],
                [pooled, diamond_y + 0.2],
                [ci[1], diamond_y],
                [pooled, diamond_y - 0.2],
            ],
            color="red",
            alpha=0.7,
        )
        ax.add_patch(diamond)

        # Formatting
        ax.axvline(0, color="gray", linestyle="-", alpha=0.5)
        ax.set_ylim(-0.5, n + 0.5)
        ax.set_ylabel("Studies")
        ax.set_xlabel("Effect Size")
        ax.set_title(f"Forest Plot - {meta_result['method'].replace('_', ' ').title()}")

        # Add heterogeneity stats
        het_text = f"I² = {meta_result['heterogeneity']['i_squared']:.1%}, "
        het_text += f"p = {meta_result['heterogeneity']['p_value']:.3f}"
        ax.text(
            0.02,
            0.98,
            het_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_yticks([])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Enhanced Bayesian A/B Testing")
    print("=" * 60)

    # Generate synthetic network data
    np.random.seed(42)
    n_users = 1000

    # Create random network
    edges = []
    for _ in range(2000):
        u1, u2 = np.random.choice(n_users, size=2, replace=False)
        edges.append((f"user_{u1}", f"user_{u2}"))

    # Generate experiment data
    data = pd.DataFrame(
        {
            "user": [f"user_{i}" for i in range(n_users)],
            "treatment": np.random.binomial(1, 0.5, n_users),
            "outcome": np.random.binomial(1, 0.1, n_users),
            "time": pd.date_range("2024-01-01", periods=n_users, freq="H"),
        }
    )

    # Test network effects
    print("\nNetwork Effect Testing:")
    network_test = NetworkEffectBayesianTest()
    network_result = network_test.test_with_network_effects(
        data, "outcome", "treatment", "user", network_edges=edges
    )

    print(f"  Direct effect: {network_result.direct_effect:.3f}")
    print(f"  Indirect effect: {network_result.indirect_effect:.3f}")
    print(f"  Total effect: {network_result.total_effect:.3f}")
    print(f"  Spillover ratio: {network_result.spillover_ratio:.3f}")
    print(f"  Network stats: {network_result.network_statistics}")

    # Test temporal effects
    print("\nTime-Dependent Testing:")
    time_test = TimeDependentBayesianTest()
    time_result = time_test.test_with_temporal_effects(
        data, "outcome", "treatment", "time", granularity="daily"
    )

    print(f"  Cumulative effect: {time_result.cumulative_effect:.3f}")
    print(f"  Peak effect: {time_result.peak_effect[1]:.3f}")
    print(f"  Decay rate: {time_result.decay_rate}")
    print(f"  Seasonality: {time_result.seasonality_detected}")
    print(f"  Trend: {time_result.trend_component}")

    # Meta-analysis example
    print("\nMeta-Analysis:")
    meta = MetaAnalysisBayesian()

    experiments = [
        {"name": "Exp 1", "effect": 0.05, "variance": 0.01, "n": 1000},
        {"name": "Exp 2", "effect": 0.08, "variance": 0.015, "n": 800},
        {"name": "Exp 3", "effect": 0.03, "variance": 0.008, "n": 1200},
        {"name": "Exp 4", "effect": 0.06, "variance": 0.012, "n": 900},
    ]

    meta_result = meta.combine_experiments(experiments, method="random_effects")
    print(f"  Pooled effect: {meta_result['pooled_effect']:.3f}")
    print(f"  95% CI: {meta_result['confidence_interval']}")
    print(f"  P-value: {meta_result['p_value']:.4f}")
    print(f"  I-squared: {meta_result['heterogeneity']['i_squared']:.1%}")
