"""
Time Series Utilities Module

This module provides utility functions for time series analysis, including:
- Data preprocessing and validation
- Feature engineering
- Statistical tests
- Visualization helpers
- Model evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict, Optional, Any
import warnings
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ============================================================================
# Data Preprocessing Utilities
# ============================================================================


def validate_time_series(
    data: Union[pd.Series, pd.DataFrame], date_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Validate and prepare time series data.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Time series data
    date_column : str, optional
        Name of the date column if DataFrame

    Returns:
    --------
    pd.DataFrame : Validated time series data
    """
    # Convert to DataFrame if Series
    if isinstance(data, pd.Series):
        df = data.to_frame(name="value")
    else:
        df = data.copy()

    # Ensure datetime index
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert index to datetime
        try:
            df.index = pd.to_datetime(df.index)
        except:
            # Create a default datetime index
            df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
            warnings.warn(
                "Could not parse datetime index. Using default daily frequency."
            )

    # Sort by index
    df.sort_index(inplace=True)

    # Check for duplicates
    if df.index.duplicated().any():
        warnings.warn("Duplicate timestamps found. Keeping first occurrence.")
        df = df[~df.index.duplicated(keep="first")]

    return df


def handle_missing_values(
    data: pd.DataFrame, method: str = "interpolate", **kwargs
) -> pd.DataFrame:
    """
    Handle missing values in time series data.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    method : str
        Method for handling missing values ('interpolate', 'forward_fill',
        'backward_fill', 'mean', 'median', 'drop')
    **kwargs : additional arguments for the method

    Returns:
    --------
    pd.DataFrame : Data with missing values handled
    """
    df = data.copy()

    if method == "interpolate":
        df = df.interpolate(method="time", **kwargs)
    elif method == "forward_fill":
        df = df.fillna(method="ffill", **kwargs)
    elif method == "backward_fill":
        df = df.fillna(method="bfill", **kwargs)
    elif method == "mean":
        df = df.fillna(df.mean(), **kwargs)
    elif method == "median":
        df = df.fillna(df.median(), **kwargs)
    elif method == "drop":
        df = df.dropna(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


def remove_outliers(
    data: pd.DataFrame, method: str = "iqr", threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from time series data.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    method : str
        Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
    threshold : float
        Threshold for outlier detection

    Returns:
    --------
    pd.DataFrame : Data with outliers removed
    """
    df = data.copy()

    for column in df.select_dtypes(include=[np.number]).columns:
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        elif method == "zscore":
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            df = df[z_scores < threshold]

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(df[column].values.reshape(-1, 1))
            df = df[outliers == 1]

    return df


def resample_time_series(
    data: pd.DataFrame, freq: str, agg_func: str = "mean"
) -> pd.DataFrame:
    """
    Resample time series to a different frequency.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with DatetimeIndex
    freq : str
        Target frequency ('D', 'W', 'M', 'Q', 'Y', 'H', etc.)
    agg_func : str
        Aggregation function ('mean', 'sum', 'max', 'min', 'median')

    Returns:
    --------
    pd.DataFrame : Resampled time series
    """
    agg_funcs = {
        "mean": np.mean,
        "sum": np.sum,
        "max": np.max,
        "min": np.min,
        "median": np.median,
    }

    if agg_func not in agg_funcs:
        raise ValueError(f"Unknown aggregation function: {agg_func}")

    return data.resample(freq).agg(agg_funcs[agg_func])


# ============================================================================
# Feature Engineering Utilities
# ============================================================================


def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from datetime index.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with DatetimeIndex

    Returns:
    --------
    pd.DataFrame : Data with additional time features
    """
    df = data.copy()

    # Basic time features
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["dayofyear"] = df.index.dayofyear
    df["weekofyear"] = df.index.isocalendar().week

    # Cyclical features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Binary features
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df.index.is_month_start.astype(int)
    df["is_month_end"] = df.index.is_month_end.astype(int)
    df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
    df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

    return df


def create_lag_features(
    data: pd.DataFrame, target_col: str, lags: List[int]
) -> pd.DataFrame:
    """
    Create lag features for time series.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    target_col : str
        Name of the target column
    lags : List[int]
        List of lag periods to create

    Returns:
    --------
    pd.DataFrame : Data with lag features
    """
    df = data.copy()

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


def create_rolling_features(
    data: pd.DataFrame,
    target_col: str,
    windows: List[int],
    functions: List[str] = ["mean", "std"],
) -> pd.DataFrame:
    """
    Create rolling window features.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    target_col : str
        Name of the target column
    windows : List[int]
        List of window sizes
    functions : List[str]
        List of aggregation functions ('mean', 'std', 'min', 'max', 'sum')

    Returns:
    --------
    pd.DataFrame : Data with rolling features
    """
    df = data.copy()

    for window in windows:
        for func in functions:
            if func == "mean":
                df[f"{target_col}_rolling_mean_{window}"] = (
                    df[target_col].rolling(window).mean()
                )
            elif func == "std":
                df[f"{target_col}_rolling_std_{window}"] = (
                    df[target_col].rolling(window).std()
                )
            elif func == "min":
                df[f"{target_col}_rolling_min_{window}"] = (
                    df[target_col].rolling(window).min()
                )
            elif func == "max":
                df[f"{target_col}_rolling_max_{window}"] = (
                    df[target_col].rolling(window).max()
                )
            elif func == "sum":
                df[f"{target_col}_rolling_sum_{window}"] = (
                    df[target_col].rolling(window).sum()
                )

    return df


def create_expanding_features(
    data: pd.DataFrame, target_col: str, functions: List[str] = ["mean", "std"]
) -> pd.DataFrame:
    """
    Create expanding window features.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    target_col : str
        Name of the target column
    functions : List[str]
        List of aggregation functions

    Returns:
    --------
    pd.DataFrame : Data with expanding features
    """
    df = data.copy()

    for func in functions:
        if func == "mean":
            df[f"{target_col}_expanding_mean"] = df[target_col].expanding().mean()
        elif func == "std":
            df[f"{target_col}_expanding_std"] = df[target_col].expanding().std()
        elif func == "min":
            df[f"{target_col}_expanding_min"] = df[target_col].expanding().min()
        elif func == "max":
            df[f"{target_col}_expanding_max"] = df[target_col].expanding().max()
        elif func == "sum":
            df[f"{target_col}_expanding_sum"] = df[target_col].expanding().sum()

    return df


# ============================================================================
# Statistical Test Utilities
# ============================================================================


def test_stationarity(
    data: pd.Series, significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Test for stationarity using multiple tests.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    significance_level : float
        Significance level for tests

    Returns:
    --------
    dict : Test results
    """
    results = {}

    # ADF Test
    adf_result = adfuller(data.dropna())
    results["adf"] = {
        "statistic": adf_result[0],
        "p_value": adf_result[1],
        "critical_values": adf_result[4],
        "is_stationary": adf_result[1] < significance_level,
    }

    # KPSS Test
    kpss_result = kpss(data.dropna())
    results["kpss"] = {
        "statistic": kpss_result[0],
        "p_value": kpss_result[1],
        "critical_values": kpss_result[3],
        "is_stationary": kpss_result[1] > significance_level,
    }

    # Overall conclusion
    results["conclusion"] = {
        "is_stationary": results["adf"]["is_stationary"]
        and results["kpss"]["is_stationary"],
        "recommendation": "Data is stationary"
        if results["adf"]["is_stationary"] and results["kpss"]["is_stationary"]
        else "Data is non-stationary, consider differencing",
    }

    return results


def test_white_noise(data: pd.Series, lags: int = 10) -> Dict[str, Any]:
    """
    Test if time series is white noise using Ljung-Box test.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    lags : int
        Number of lags to test

    Returns:
    --------
    dict : Test results
    """
    lb_test = acorr_ljungbox(data.dropna(), lags=lags, return_df=True)

    return {
        "statistics": lb_test["lb_stat"].tolist(),
        "p_values": lb_test["lb_pvalue"].tolist(),
        "is_white_noise": all(lb_test["lb_pvalue"] > 0.05),
        "recommendation": "Series appears to be white noise"
        if all(lb_test["lb_pvalue"] > 0.05)
        else "Series has significant autocorrelation",
    }


def test_normality(data: pd.Series) -> Dict[str, Any]:
    """
    Test for normality using multiple tests.

    Parameters:
    -----------
    data : pd.Series
        Time series data

    Returns:
    --------
    dict : Test results
    """
    results = {}

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data.dropna())
    results["shapiro"] = {
        "statistic": shapiro_stat,
        "p_value": shapiro_p,
        "is_normal": shapiro_p > 0.05,
    }

    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(data.dropna())
    results["jarque_bera"] = {
        "statistic": jb_stat,
        "p_value": jb_p,
        "is_normal": jb_p > 0.05,
    }

    # Skewness and Kurtosis
    results["skewness"] = stats.skew(data.dropna())
    results["kurtosis"] = stats.kurtosis(data.dropna())

    # Overall conclusion
    results["conclusion"] = {
        "is_normal": results["shapiro"]["is_normal"]
        and results["jarque_bera"]["is_normal"],
        "recommendation": "Data appears normally distributed"
        if results["shapiro"]["is_normal"] and results["jarque_bera"]["is_normal"]
        else "Data is not normally distributed, consider transformation",
    }

    return results


# ============================================================================
# Model Evaluation Metrics
# ============================================================================


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values

    Returns:
    --------
    dict : Evaluation metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    # Basic metrics
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)

    # Percentage errors
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape = (
        np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))
        * 100
    )

    # Directional accuracy
    direction_actual = np.diff(actual) > 0
    direction_predicted = np.diff(predicted) > 0
    directional_accuracy = np.mean(direction_actual == direction_predicted) * 100

    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Mean Absolute Scaled Error (MASE)
    naive_forecast = actual[:-1]  # Use previous value as naive forecast
    naive_errors = actual[1:] - naive_forecast
    mase = mae / np.mean(np.abs(naive_errors))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "directional_accuracy": directional_accuracy,
        "r2": r2,
        "mase": mase,
    }


def calculate_residual_diagnostics(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Perform comprehensive residual diagnostics.

    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals

    Returns:
    --------
    dict : Diagnostic results
    """
    results = {}

    # Basic statistics
    results["mean"] = np.mean(residuals)
    results["std"] = np.std(residuals)
    results["skewness"] = stats.skew(residuals)
    results["kurtosis"] = stats.kurtosis(residuals)

    # Normality test
    results["normality"] = test_normality(pd.Series(residuals))

    # Autocorrelation
    results["autocorrelation"] = test_white_noise(pd.Series(residuals))

    # Heteroscedasticity (simplified Breusch-Pagan test concept)
    x = np.arange(len(residuals))
    correlation = np.corrcoef(x, residuals**2)[0, 1]
    results["heteroscedasticity"] = {
        "correlation": correlation,
        "appears_heteroscedastic": abs(correlation) > 0.3,
    }

    return results


# ============================================================================
# Visualization Utilities
# ============================================================================


def plot_time_series_decomposition(
    data: pd.Series, period: Optional[int] = None, model: str = "additive"
):
    """
    Plot time series decomposition.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    period : int, optional
        Seasonal period
    model : str
        Decomposition model ('additive' or 'multiplicative')
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    if period is None:
        # Try to infer period
        if hasattr(data.index, "freq"):
            if data.index.freq == "D":
                period = 7  # Weekly seasonality for daily data
            elif data.index.freq == "M":
                period = 12  # Yearly seasonality for monthly data
            else:
                period = 4  # Default
        else:
            period = 4

    decomposition = seasonal_decompose(data, model=model, period=period)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    data.plot(ax=axes[0], title="Original Time Series")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
    decomposition.resid.plot(ax=axes[3], title="Residual")

    plt.tight_layout()
    return fig


def plot_acf_pacf(data: pd.Series, lags: int = 40):
    """
    Plot ACF and PACF.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    lags : int
        Number of lags to plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # ACF
    acf_values = acf(data.dropna(), nlags=lags)
    axes[0].bar(range(len(acf_values)), acf_values)
    axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[0].axhline(
        y=1.96 / np.sqrt(len(data)), color="r", linestyle="--", linewidth=0.5
    )
    axes[0].axhline(
        y=-1.96 / np.sqrt(len(data)), color="r", linestyle="--", linewidth=0.5
    )
    axes[0].set_title("Autocorrelation Function (ACF)")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Correlation")

    # PACF
    pacf_values = pacf(data.dropna(), nlags=lags)
    axes[1].bar(range(len(pacf_values)), pacf_values)
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[1].axhline(
        y=1.96 / np.sqrt(len(data)), color="r", linestyle="--", linewidth=0.5
    )
    axes[1].axhline(
        y=-1.96 / np.sqrt(len(data)), color="r", linestyle="--", linewidth=0.5
    )
    axes[1].set_title("Partial Autocorrelation Function (PACF)")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Correlation")

    plt.tight_layout()
    return fig


def plot_forecast_with_intervals(
    actual: pd.Series,
    forecast: pd.Series,
    lower_bound: Optional[pd.Series] = None,
    upper_bound: Optional[pd.Series] = None,
    title: str = "Forecast vs Actual",
):
    """
    Plot forecast with confidence intervals.

    Parameters:
    -----------
    actual : pd.Series
        Actual values
    forecast : pd.Series
        Forecasted values
    lower_bound : pd.Series, optional
        Lower confidence bound
    upper_bound : pd.Series, optional
        Upper confidence bound
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual values
    actual.plot(ax=ax, label="Actual", color="blue", linewidth=2)

    # Plot forecast
    forecast.plot(ax=ax, label="Forecast", color="red", linewidth=2)

    # Plot confidence intervals if provided
    if lower_bound is not None and upper_bound is not None:
        ax.fill_between(
            forecast.index,
            lower_bound,
            upper_bound,
            color="red",
            alpha=0.2,
            label="Confidence Interval",
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# Cross-Validation Utilities
# ============================================================================


def time_series_split(
    data: pd.DataFrame, n_splits: int = 5, test_size: Optional[int] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create time series train/test splits for cross-validation.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    n_splits : int
        Number of splits
    test_size : int, optional
        Size of test set (if None, computed automatically)

    Returns:
    --------
    list : List of (train, test) tuples
    """
    splits = []

    if test_size is None:
        test_size = len(data) // (n_splits + 1)

    for i in range(n_splits):
        # Calculate split point
        split_point = len(data) - test_size * (n_splits - i)

        if split_point > test_size:  # Ensure we have enough training data
            train = data.iloc[:split_point]
            test = data.iloc[split_point : split_point + test_size]
            splits.append((train, test))

    return splits


def walk_forward_validation(
    data: pd.DataFrame,
    model_func,
    initial_train_size: int,
    step_size: int = 1,
    horizon: int = 1,
) -> Dict[str, Any]:
    """
    Perform walk-forward validation.

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    model_func : callable
        Function that trains model and returns predictions
    initial_train_size : int
        Initial training set size
    step_size : int
        Step size for moving forward
    horizon : int
        Forecast horizon

    Returns:
    --------
    dict : Validation results
    """
    predictions = []
    actuals = []

    for i in range(initial_train_size, len(data) - horizon + 1, step_size):
        # Split data
        train = data.iloc[:i]
        test = data.iloc[i : i + horizon]

        # Make prediction
        pred = model_func(train, horizon)

        predictions.extend(pred)
        actuals.extend(test.values.flatten())

    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    metrics = calculate_metrics(actuals, predictions)

    return {"predictions": predictions, "actuals": actuals, "metrics": metrics}


# ============================================================================
# Transformation Utilities
# ============================================================================


def box_cox_transform(data: pd.Series) -> Tuple[pd.Series, float]:
    """
    Apply Box-Cox transformation.

    Parameters:
    -----------
    data : pd.Series
        Time series data

    Returns:
    --------
    tuple : (transformed_data, lambda_value)
    """
    # Ensure positive values
    if (data <= 0).any():
        data = data - data.min() + 1

    transformed, lambda_val = stats.boxcox(data)

    return pd.Series(transformed, index=data.index), lambda_val


def inverse_box_cox(data: pd.Series, lambda_val: float) -> pd.Series:
    """
    Inverse Box-Cox transformation.

    Parameters:
    -----------
    data : pd.Series
        Transformed data
    lambda_val : float
        Lambda value from original transformation

    Returns:
    --------
    pd.Series : Original scale data
    """
    if lambda_val == 0:
        return pd.Series(np.exp(data), index=data.index)
    else:
        return pd.Series((data * lambda_val + 1) ** (1 / lambda_val), index=data.index)


def difference_series(data: pd.Series, order: int = 1) -> pd.Series:
    """
    Difference time series.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    order : int
        Differencing order

    Returns:
    --------
    pd.Series : Differenced series
    """
    return data.diff(order).dropna()


def inverse_difference(data: pd.Series, original_value: float) -> pd.Series:
    """
    Inverse differencing operation.

    Parameters:
    -----------
    data : pd.Series
        Differenced data
    original_value : float
        Original value to start cumsum from

    Returns:
    --------
    pd.Series : Original scale data
    """
    return pd.Series(np.r_[original_value, data.values].cumsum()[1:], index=data.index)


# ============================================================================
# Forecasting Utilities
# ============================================================================


def combine_forecasts(
    forecasts: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    method: str = "average",
) -> np.ndarray:
    """
    Combine multiple forecasts.

    Parameters:
    -----------
    forecasts : dict
        Dictionary of model_name: forecast_array
    weights : dict, optional
        Dictionary of model_name: weight
    method : str
        Combination method ('average', 'weighted', 'median', 'trimmed_mean')

    Returns:
    --------
    np.ndarray : Combined forecast
    """
    forecast_arrays = list(forecasts.values())

    if method == "average":
        return np.mean(forecast_arrays, axis=0)

    elif method == "weighted":
        if weights is None:
            weights = {name: 1 / len(forecasts) for name in forecasts.keys()}

        weighted_sum = np.zeros_like(forecast_arrays[0])
        for name, forecast in forecasts.items():
            weighted_sum += weights.get(name, 0) * forecast

        return weighted_sum

    elif method == "median":
        return np.median(forecast_arrays, axis=0)

    elif method == "trimmed_mean":
        # Remove highest and lowest, then average
        sorted_forecasts = np.sort(forecast_arrays, axis=0)
        if len(sorted_forecasts) > 2:
            return np.mean(sorted_forecasts[1:-1], axis=0)
        else:
            return np.mean(sorted_forecasts, axis=0)

    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_prediction_intervals(
    point_forecast: np.ndarray, residuals: np.ndarray, confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals.

    Parameters:
    -----------
    point_forecast : np.ndarray
        Point forecast values
    residuals : np.ndarray
        Model residuals
    confidence_level : float
        Confidence level for intervals

    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    # Calculate standard error
    std_error = np.std(residuals)

    # Calculate z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate intervals
    margin = z_score * std_error
    lower_bound = point_forecast - margin
    upper_bound = point_forecast + margin

    return lower_bound, upper_bound


# ============================================================================
# Utility Functions for Specific Models
# ============================================================================


def suggest_arima_order(
    data: pd.Series, max_p: int = 5, max_q: int = 5
) -> Tuple[int, int, int]:
    """
    Suggest ARIMA order based on ACF and PACF analysis.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    max_p : int
        Maximum p order to consider
    max_q : int
        Maximum q order to consider

    Returns:
    --------
    tuple : Suggested (p, d, q) order
    """
    # Test for stationarity
    stationarity = test_stationarity(data)

    # Determine d (differencing order)
    d = 0
    diff_data = data.copy()
    while not stationarity["adf"]["is_stationary"] and d < 2:
        d += 1
        diff_data = diff_data.diff().dropna()
        stationarity = test_stationarity(diff_data)

    # Analyze ACF and PACF
    acf_values = acf(diff_data.dropna(), nlags=max_q)
    pacf_values = pacf(diff_data.dropna(), nlags=max_p)

    # Suggest p (AR order) - look for PACF cutoff
    p = 0
    for i in range(1, min(len(pacf_values), max_p + 1)):
        if abs(pacf_values[i]) > 1.96 / np.sqrt(len(diff_data)):
            p = i
        else:
            break

    # Suggest q (MA order) - look for ACF cutoff
    q = 0
    for i in range(1, min(len(acf_values), max_q + 1)):
        if abs(acf_values[i]) > 1.96 / np.sqrt(len(diff_data)):
            q = i
        else:
            break

    return (p, d, q)


def detect_seasonality(data: pd.Series, max_lag: int = 50) -> Dict[str, Any]:
    """
    Detect seasonality in time series.

    Parameters:
    -----------
    data : pd.Series
        Time series data
    max_lag : int
        Maximum lag to check

    Returns:
    --------
    dict : Seasonality information
    """
    # Calculate ACF
    acf_values = acf(data.dropna(), nlags=max_lag)

    # Find peaks in ACF (potential seasonal periods)
    from scipy.signal import find_peaks

    peaks, properties = find_peaks(acf_values[1:], height=0.3, distance=2)
    peaks = peaks + 1  # Adjust for removed first element

    results = {
        "has_seasonality": len(peaks) > 0,
        "seasonal_periods": peaks.tolist(),
        "seasonal_strength": properties["peak_heights"].tolist()
        if len(peaks) > 0
        else [],
    }

    # Determine primary seasonal period
    if len(peaks) > 0:
        primary_idx = np.argmax(properties["peak_heights"])
        results["primary_period"] = peaks[primary_idx]
        results["primary_strength"] = properties["peak_heights"][primary_idx]
    else:
        results["primary_period"] = None
        results["primary_strength"] = 0

    return results


# ============================================================================
# Export all functions
# ============================================================================

__all__ = [
    # Data Preprocessing
    "validate_time_series",
    "handle_missing_values",
    "remove_outliers",
    "resample_time_series",
    # Feature Engineering
    "create_time_features",
    "create_lag_features",
    "create_rolling_features",
    "create_expanding_features",
    # Statistical Tests
    "test_stationarity",
    "test_white_noise",
    "test_normality",
    # Model Evaluation
    "calculate_metrics",
    "calculate_residual_diagnostics",
    # Visualization
    "plot_time_series_decomposition",
    "plot_acf_pacf",
    "plot_forecast_with_intervals",
    # Cross-Validation
    "time_series_split",
    "walk_forward_validation",
    # Transformations
    "box_cox_transform",
    "inverse_box_cox",
    "difference_series",
    "inverse_difference",
    # Forecasting
    "combine_forecasts",
    "calculate_prediction_intervals",
    "suggest_arima_order",
    "detect_seasonality",
]
