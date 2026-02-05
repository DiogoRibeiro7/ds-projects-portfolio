"""Time Series Feature Extraction

Advanced time series feature engineering techniques.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# Optional imports
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from tsfresh import extract_features, select_features
    from tsfresh.feature_extraction import ComprehensiveFCParameters

    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False


class TimeSeriesFeatureExtractor:
    """Comprehensive time series feature extraction."""

    def __init__(
        self,
        window_sizes: list[int] = None,
        include_statistical: bool = True,
        include_frequency: bool = True,
        include_entropy: bool = True,
        include_autocorrelation: bool = True,
        include_tsfresh: bool = False,
    ):
        """Initialize time series feature extractor.

        Args:
            window_sizes: List of window sizes for rolling features
            include_statistical: Include statistical features
            include_frequency: Include frequency domain features
            include_entropy: Include entropy-based features
            include_autocorrelation: Include autocorrelation features
            include_tsfresh: Use TSFresh for comprehensive extraction
        """
        self.window_sizes = window_sizes or [3, 5, 7, 10, 15]
        self.include_statistical = include_statistical
        self.include_frequency = include_frequency
        self.include_entropy = include_entropy
        self.include_autocorrelation = include_autocorrelation
        self.include_tsfresh = include_tsfresh and TSFRESH_AVAILABLE

    def extract_features(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_cols: list[str],
        entity_col: str | None = None,
    ) -> pd.DataFrame:
        """Extract time series features.

        Args:
            df: Input dataframe
            time_col: Time column name
            value_cols: List of value columns to extract features from
            entity_col: Optional entity column for grouped extraction

        Returns:
            DataFrame with extracted features
        """
        # Sort by time
        df = df.sort_values(time_col)

        features_list = []

        if entity_col:
            # Group by entity and extract features
            for entity in df[entity_col].unique():
                entity_df = df[df[entity_col] == entity].copy()
                entity_features = self._extract_all_features(entity_df, value_cols)
                entity_features[entity_col] = entity
                features_list.append(entity_features)

            return pd.concat(features_list, ignore_index=True)

        else:
            return self._extract_all_features(df, value_cols)

    def _extract_all_features(
        self, df: pd.DataFrame, value_cols: list[str]
    ) -> pd.DataFrame:
        """Extract all types of features."""
        features = df.copy()

        for col in value_cols:
            if col not in df.columns:
                continue

            series = df[col].values

            # Statistical features
            if self.include_statistical:
                stat_features = self._extract_statistical_features(series, col)
                for feat_name, feat_value in stat_features.items():
                    features[feat_name] = feat_value

            # Rolling features
            roll_features = self._extract_rolling_features(df[col], col)
            features = pd.concat([features, roll_features], axis=1)

            # Frequency domain features
            if self.include_frequency:
                freq_features = self._extract_frequency_features(series, col)
                for feat_name, feat_value in freq_features.items():
                    features[feat_name] = feat_value

            # Entropy features
            if self.include_entropy:
                entropy_features = self._extract_entropy_features(series, col)
                for feat_name, feat_value in entropy_features.items():
                    features[feat_name] = feat_value

            # Autocorrelation features
            if self.include_autocorrelation and STATSMODELS_AVAILABLE:
                autocorr_features = self._extract_autocorrelation_features(series, col)
                for feat_name, feat_value in autocorr_features.items():
                    features[feat_name] = feat_value

            # Lag features
            lag_features = self._extract_lag_features(df[col], col)
            features = pd.concat([features, lag_features], axis=1)

        # TSFresh features
        if self.include_tsfresh and TSFRESH_AVAILABLE:
            tsfresh_features = self._extract_tsfresh_features(df, value_cols)
            features = pd.concat([features, tsfresh_features], axis=1)

        return features

    def _extract_statistical_features(
        self, series: np.ndarray, prefix: str
    ) -> dict[str, float]:
        """Extract basic statistical features."""
        features = {}

        # Handle NaN values
        valid_series = series[~np.isnan(series)]

        if len(valid_series) == 0:
            return features

        # Basic statistics
        features[f"{prefix}_mean"] = np.mean(valid_series)
        features[f"{prefix}_std"] = np.std(valid_series)
        features[f"{prefix}_var"] = np.var(valid_series)
        features[f"{prefix}_min"] = np.min(valid_series)
        features[f"{prefix}_max"] = np.max(valid_series)
        features[f"{prefix}_range"] = np.max(valid_series) - np.min(valid_series)
        features[f"{prefix}_median"] = np.median(valid_series)

        # Percentiles
        for p in [25, 75, 90, 95]:
            features[f"{prefix}_p{p}"] = np.percentile(valid_series, p)

        # Higher moments
        features[f"{prefix}_skew"] = stats.skew(valid_series)
        features[f"{prefix}_kurtosis"] = stats.kurtosis(valid_series)

        # Coefficient of variation
        if features[f"{prefix}_mean"] != 0:
            features[f"{prefix}_cv"] = features[f"{prefix}_std"] / abs(
                features[f"{prefix}_mean"]
            )

        # Peak to peak
        features[f"{prefix}_peak_to_peak"] = np.ptp(valid_series)

        # Mean absolute deviation
        features[f"{prefix}_mad"] = np.mean(
            np.abs(valid_series - features[f"{prefix}_mean"])
        )

        # Root mean square
        features[f"{prefix}_rms"] = np.sqrt(np.mean(valid_series**2))

        return features

    def _extract_rolling_features(self, series: pd.Series, prefix: str) -> pd.DataFrame:
        """Extract rolling window features."""
        rolling_dfs = []

        for window in self.window_sizes:
            # Rolling statistics
            roll = series.rolling(window=window, min_periods=1)

            roll_df = pd.DataFrame(
                {
                    f"{prefix}_roll_mean_{window}": roll.mean(),
                    f"{prefix}_roll_std_{window}": roll.std(),
                    f"{prefix}_roll_min_{window}": roll.min(),
                    f"{prefix}_roll_max_{window}": roll.max(),
                    f"{prefix}_roll_median_{window}": roll.median(),
                    f"{prefix}_roll_skew_{window}": roll.skew(),
                    f"{prefix}_roll_kurt_{window}": roll.kurt(),
                }
            )

            # Rolling range
            roll_df[f"{prefix}_roll_range_{window}"] = (
                roll_df[f"{prefix}_roll_max_{window}"]
                - roll_df[f"{prefix}_roll_min_{window}"]
            )

            # Rolling coefficient of variation
            roll_df[f"{prefix}_roll_cv_{window}"] = roll_df[
                f"{prefix}_roll_std_{window}"
            ] / (roll_df[f"{prefix}_roll_mean_{window}"] + 1e-8)

            rolling_dfs.append(roll_df)

        return pd.concat(rolling_dfs, axis=1)

    def _extract_frequency_features(
        self, series: np.ndarray, prefix: str
    ) -> dict[str, float]:
        """Extract frequency domain features."""
        features = {}

        # Handle NaN values
        valid_series = series[~np.isnan(series)]

        if len(valid_series) < 2:
            return features

        # Compute FFT
        fft_values = np.abs(fft(valid_series))
        freqs = fftfreq(len(valid_series))

        # Only positive frequencies
        pos_mask = freqs > 0
        fft_pos = fft_values[pos_mask]
        freqs_pos = freqs[pos_mask]

        if len(fft_pos) == 0:
            return features

        # Spectral features
        features[f"{prefix}_spectral_mean"] = np.mean(fft_pos)
        features[f"{prefix}_spectral_std"] = np.std(fft_pos)
        features[f"{prefix}_spectral_max"] = np.max(fft_pos)

        # Dominant frequency
        dominant_idx = np.argmax(fft_pos)
        features[f"{prefix}_dominant_freq"] = freqs_pos[dominant_idx]
        features[f"{prefix}_dominant_freq_magnitude"] = fft_pos[dominant_idx]

        # Spectral entropy
        if np.sum(fft_pos) > 0:
            fft_normalized = fft_pos / np.sum(fft_pos)
            features[f"{prefix}_spectral_entropy"] = entropy(fft_normalized)

        # Spectral energy
        features[f"{prefix}_spectral_energy"] = np.sum(fft_pos**2)

        # Band power
        total_power = np.sum(fft_pos**2)
        if total_power > 0:
            # Low frequency band (0-25% of frequencies)
            low_band_idx = int(len(fft_pos) * 0.25)
            features[f"{prefix}_low_band_power"] = (
                np.sum(fft_pos[:low_band_idx] ** 2) / total_power
            )

            # Mid frequency band (25-75% of frequencies)
            mid_band_idx = int(len(fft_pos) * 0.75)
            features[f"{prefix}_mid_band_power"] = (
                np.sum(fft_pos[low_band_idx:mid_band_idx] ** 2) / total_power
            )

            # High frequency band (75-100% of frequencies)
            features[f"{prefix}_high_band_power"] = (
                np.sum(fft_pos[mid_band_idx:] ** 2) / total_power
            )

        return features

    def _extract_entropy_features(
        self, series: np.ndarray, prefix: str
    ) -> dict[str, float]:
        """Extract entropy-based features."""
        features = {}

        # Handle NaN values
        valid_series = series[~np.isnan(series)]

        if len(valid_series) < 2:
            return features

        # Shannon entropy (discretized)
        hist, _ = np.histogram(valid_series, bins=10)
        hist = hist[hist > 0]  # Remove zero bins
        if len(hist) > 0:
            prob = hist / np.sum(hist)
            features[f"{prefix}_shannon_entropy"] = entropy(prob)

        # Approximate entropy
        features[f"{prefix}_approx_entropy"] = self._approximate_entropy(valid_series)

        # Sample entropy
        features[f"{prefix}_sample_entropy"] = self._sample_entropy(valid_series)

        # Permutation entropy
        features[f"{prefix}_permutation_entropy"] = self._permutation_entropy(
            valid_series
        )

        return features

    def _approximate_entropy(
        self, series: np.ndarray, m: int = 2, r: float = 0.2
    ) -> float:
        """Calculate approximate entropy."""
        N = len(series)
        if N < m + 1:
            return 0

        def _maxdist(x_i, x_j, m):
            return max([abs(float(x_i[k]) - float(x_j[k])) for k in range(m)])

        def _phi(m):
            templates = np.array([series[i : i + m] for i in range(N - m + 1)])
            C = [0] * (N - m + 1)
            for i in range(N - m + 1):
                template = templates[i]
                for j in range(N - m + 1):
                    if _maxdist(template, templates[j], m) <= r:
                        C[i] += 1
            phi = sum([np.log(c / (N - m + 1)) for c in C]) / (N - m + 1)
            return phi

        return _phi(m) - _phi(m + 1)

    def _sample_entropy(self, series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy."""
        N = len(series)
        if N < m + 1:
            return 0

        def _maxdist(x_i, x_j, m):
            return max([abs(float(x_i[k]) - float(x_j[k])) for k in range(m)])

        def _phi(m):
            templates = np.array([series[i : i + m] for i in range(N - m + 1)])
            C = 0
            for i in range(N - m):
                for j in range(i + 1, N - m + 1):
                    if _maxdist(templates[i], templates[j], m) <= r:
                        C += 1
            return C

        B = _phi(m)
        A = _phi(m + 1)

        if B == 0:
            return 0

        return -np.log(A / B)

    def _permutation_entropy(
        self, series: np.ndarray, order: int = 3, delay: int = 1
    ) -> float:
        """Calculate permutation entropy."""
        N = len(series)
        if N < order:
            return 0

        # Create permutations
        permutations = []
        for i in range(N - (order - 1) * delay):
            indices = [i + j * delay for j in range(order)]
            permutation = tuple(np.argsort([series[idx] for idx in indices]))
            permutations.append(permutation)

        # Count permutation frequencies
        from collections import Counter

        perm_counts = Counter(permutations)
        total = len(permutations)

        # Calculate entropy
        probs = np.array(list(perm_counts.values())) / total
        return entropy(probs)

    def _extract_autocorrelation_features(
        self, series: np.ndarray, prefix: str
    ) -> dict[str, float]:
        """Extract autocorrelation features."""
        features = {}

        if not STATSMODELS_AVAILABLE:
            return features

        # Handle NaN values
        valid_series = series[~np.isnan(series)]

        if len(valid_series) < 10:
            return features

        # Autocorrelation
        max_lag = min(20, len(valid_series) // 4)
        acf_values = acf(valid_series, nlags=max_lag)

        # First significant lag
        for i, val in enumerate(acf_values[1:], 1):
            if abs(val) < 0.1:  # Threshold for significance
                features[f"{prefix}_first_zero_crossing"] = i
                break

        # Autocorrelation at specific lags
        for lag in [1, 2, 3, 5, 10]:
            if lag < len(acf_values):
                features[f"{prefix}_acf_lag_{lag}"] = acf_values[lag]

        # Partial autocorrelation
        try:
            pacf_values = pacf(valid_series, nlags=min(10, max_lag))
            for lag in [1, 2, 3, 5]:
                if lag < len(pacf_values):
                    features[f"{prefix}_pacf_lag_{lag}"] = pacf_values[lag]
        except:
            pass

        # Stationarity tests
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(valid_series, autolag="AIC")
            features[f"{prefix}_adf_statistic"] = adf_result[0]
            features[f"{prefix}_adf_pvalue"] = adf_result[1]
            features[f"{prefix}_is_stationary_adf"] = float(adf_result[1] < 0.05)
        except:
            pass

        try:
            # KPSS test
            kpss_result = kpss(valid_series, regression="c")
            features[f"{prefix}_kpss_statistic"] = kpss_result[0]
            features[f"{prefix}_kpss_pvalue"] = kpss_result[1]
            features[f"{prefix}_is_stationary_kpss"] = float(kpss_result[1] > 0.05)
        except:
            pass

        return features

    def _extract_lag_features(
        self, series: pd.Series, prefix: str, lags: list[int] = None
    ) -> pd.DataFrame:
        """Extract lag features."""
        if lags is None:
            lags = [1, 2, 3, 5, 7, 10, 15, 30]

        lag_dfs = []

        for lag in lags:
            # Lag values
            lag_df = pd.DataFrame({f"{prefix}_lag_{lag}": series.shift(lag)})

            # Difference from lag
            lag_df[f"{prefix}_diff_lag_{lag}"] = series - series.shift(lag)

            # Ratio to lag
            lag_df[f"{prefix}_ratio_lag_{lag}"] = series / (series.shift(lag) + 1e-8)

            # Percentage change from lag
            lag_df[f"{prefix}_pct_change_lag_{lag}"] = series.pct_change(periods=lag)

            lag_dfs.append(lag_df)

        return pd.concat(lag_dfs, axis=1)

    def _extract_tsfresh_features(
        self, df: pd.DataFrame, value_cols: list[str]
    ) -> pd.DataFrame:
        """Extract features using TSFresh."""
        if not TSFRESH_AVAILABLE:
            return pd.DataFrame()

        try:
            # Prepare data for TSFresh
            tsfresh_df = df.copy()
            tsfresh_df["id"] = 0  # Single time series

            # Extract features
            extraction_settings = ComprehensiveFCParameters()
            features = extract_features(
                tsfresh_df,
                column_id="id",
                column_value=value_cols[0] if len(value_cols) == 1 else None,
                default_fc_parameters=extraction_settings,
                disable_progressbar=True,
            )

            # Flatten and rename columns
            features.columns = [f"tsfresh_{col}" for col in features.columns]

            # Repeat for all rows (since TSFresh aggregates)
            features = pd.concat([features] * len(df), ignore_index=True)

            return features

        except Exception as e:
            print(f"TSFresh extraction failed: {e}")
            return pd.DataFrame()


class SeasonalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract seasonal features from time series."""

    def __init__(self, period: int = 12, method: str = "decompose"):
        """Initialize seasonal feature extractor.

        Args:
            period: Seasonal period
            method: 'decompose', 'stl', 'fourier'
        """
        self.period = period
        self.method = method

    def fit(self, X, y=None):
        """Fit is not needed for seasonal extraction."""
        return self

    def transform(self, X):
        """Extract seasonal features."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        for col in X_df.columns:
            if self.method == "decompose" and STATSMODELS_AVAILABLE:
                try:
                    # Seasonal decomposition
                    decomposition = seasonal_decompose(
                        X_df[col],
                        model="additive",
                        period=self.period,
                        extrapolate_trend="freq",
                    )

                    X_df[f"{col}_trend"] = decomposition.trend
                    X_df[f"{col}_seasonal"] = decomposition.seasonal
                    X_df[f"{col}_residual"] = decomposition.resid

                except Exception as e:
                    print(f"Seasonal decomposition failed for {col}: {e}")

            elif self.method == "fourier":
                # Fourier features for seasonality
                for k in range(1, min(4, self.period // 2)):
                    X_df[f"{col}_sin_{k}"] = np.sin(
                        2 * np.pi * k * np.arange(len(X_df)) / self.period
                    )
                    X_df[f"{col}_cos_{k}"] = np.cos(
                        2 * np.pi * k * np.arange(len(X_df)) / self.period
                    )

        return X_df


class WindowFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features using sliding windows."""

    def __init__(
        self,
        window_size: int = 10,
        step_size: int = 1,
        feature_functions: list[callable] = None,
    ):
        """Initialize window feature extractor.

        Args:
            window_size: Size of sliding window
            step_size: Step size for sliding
            feature_functions: List of functions to apply to windows
        """
        self.window_size = window_size
        self.step_size = step_size
        self.feature_functions = feature_functions or [
            np.mean,
            np.std,
            np.min,
            np.max,
            np.median,
        ]

    def fit(self, X, y=None):
        """Fit is not needed."""
        return self

    def transform(self, X):
        """Extract window features."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        features = []

        for col in X_df.columns:
            series = X_df[col].values

            # Extract windows
            windows = []
            for i in range(0, len(series) - self.window_size + 1, self.step_size):
                window = series[i : i + self.window_size]
                windows.append(window)

            # Apply functions to windows
            for func in self.feature_functions:
                func_name = func.__name__
                window_features = [func(window) for window in windows]

                # Pad to match original length
                padded = np.full(len(series), np.nan)
                for i, val in enumerate(window_features):
                    padded[i * self.step_size] = val

                # Forward fill
                padded_series = (
                    pd.Series(padded).fillna(method="ffill").fillna(method="bfill")
                )
                features.append(padded_series.values)

        # Create feature names
        feature_names = []
        for col in X_df.columns:
            for func in self.feature_functions:
                feature_names.append(f"{col}_window_{func.__name__}")

        # Create dataframe
        features_df = pd.DataFrame(
            np.column_stack(features), columns=feature_names, index=X_df.index
        )

        return pd.concat([X_df, features_df], axis=1)


class ChangePointDetector(BaseEstimator, TransformerMixin):
    """Detect and create features from change points in time series."""

    def __init__(self, method: str = "cusum", threshold: float = 0.05):
        """Initialize change point detector.

        Args:
            method: 'cusum', 'pelt', 'binary_segmentation'
            threshold: Threshold for detection
        """
        self.method = method
        self.threshold = threshold
        self.change_points = {}

    def fit(self, X, y=None):
        """Detect change points."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        for col in X_df.columns:
            if self.method == "cusum":
                change_points = self._detect_cusum(X_df[col].values)
            elif self.method == "binary_segmentation":
                change_points = self._detect_binary_segmentation(X_df[col].values)
            else:
                change_points = []

            self.change_points[col] = change_points

        return self

    def transform(self, X):
        """Create features from change points."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        for col in X_df.columns:
            if col in self.change_points:
                # Distance to nearest change point
                distances = np.full(len(X_df), np.inf)
                for cp in self.change_points[col]:
                    distances = np.minimum(distances, np.abs(np.arange(len(X_df)) - cp))

                X_df[f"{col}_dist_to_changepoint"] = distances

                # Is change point
                X_df[f"{col}_is_changepoint"] = 0
                X_df.loc[self.change_points[col], f"{col}_is_changepoint"] = 1

                # Segment ID
                segment_id = np.zeros(len(X_df))
                current_segment = 0
                for i in range(len(X_df)):
                    if i in self.change_points[col]:
                        current_segment += 1
                    segment_id[i] = current_segment

                X_df[f"{col}_segment"] = segment_id

        return X_df

    def _detect_cusum(self, series: np.ndarray) -> list[int]:
        """Detect change points using CUSUM."""
        mean = np.mean(series)
        std = np.std(series)
        threshold = self.threshold * std

        cusum_pos = np.zeros(len(series))
        cusum_neg = np.zeros(len(series))
        change_points = []

        for i in range(1, len(series)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + series[i] - mean - threshold)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - series[i] + mean - threshold)

            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                change_points.append(i)
                cusum_pos[i] = 0
                cusum_neg[i] = 0

        return change_points

    def _detect_binary_segmentation(self, series: np.ndarray) -> list[int]:
        """Detect change points using binary segmentation."""
        change_points = []

        def segment(start, end):
            if end - start < 10:  # Minimum segment size
                return

            # Find best split point
            best_gain = -np.inf
            best_split = None

            for split in range(start + 5, end - 5):
                left_mean = np.mean(series[start:split])
                right_mean = np.mean(series[split:end])

                left_var = np.var(series[start:split])
                right_var = np.var(series[split:end])

                # Calculate gain
                gain = (split - start) * left_var + (end - split) * right_var
                total_var = np.var(series[start:end]) * (end - start)
                gain = total_var - gain

                if gain > best_gain:
                    best_gain = gain
                    best_split = split

            # Check if gain is significant
            if best_gain > self.threshold * np.var(series[start:end]) * (end - start):
                change_points.append(best_split)
                segment(start, best_split)
                segment(best_split, end)

        segment(0, len(series))
        return sorted(change_points)
