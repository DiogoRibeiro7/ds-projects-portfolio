"""
Custom exception classes for the data science portfolio.

This module provides specific exception types for better error handling
and debugging across the codebase.
"""

from typing import Any, Dict, Optional


class PortfolioBaseException(Exception):
    """Base exception class for all portfolio-specific exceptions."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the base exception.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.error_code}: {self.message} | Details: {self.details}"
        return f"{self.error_code}: {self.message}"


# ============== Data Exceptions ==============


class DataValidationError(PortfolioBaseException):
    """Raised when data validation fails."""

    pass


class MissingDataError(PortfolioBaseException):
    """Raised when required data is missing."""

    pass


class DataQualityError(PortfolioBaseException):
    """Raised when data quality checks fail."""

    pass


class InsufficientDataError(PortfolioBaseException):
    """Raised when there's not enough data for analysis."""

    def __init__(
        self,
        message: str,
        required: int,
        actual: int,
        **kwargs: Any,
    ) -> None:
        """Initialize with sample size information."""
        details = {"required_samples": required, "actual_samples": actual}
        super().__init__(message, details=details, **kwargs)


class DataTypeError(PortfolioBaseException):
    """Raised when data types don't match expectations."""

    def __init__(
        self,
        message: str,
        expected_type: str,
        actual_type: str,
        column: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with type information."""
        details = {
            "expected_type": expected_type,
            "actual_type": actual_type,
        }
        if column:
            details["column"] = column
        super().__init__(message, details=details, **kwargs)


# ============== Statistical Exceptions ==============


class StatisticalError(PortfolioBaseException):
    """Base class for statistical errors."""

    pass


class InsufficientPowerError(StatisticalError):
    """Raised when statistical power is too low."""

    def __init__(
        self,
        message: str,
        actual_power: float,
        required_power: float = 0.80,
        **kwargs: Any,
    ) -> None:
        """Initialize with power information."""
        details = {
            "actual_power": actual_power,
            "required_power": required_power,
            "power_gap": required_power - actual_power,
        }
        super().__init__(message, details=details, **kwargs)


class SampleRatioMismatchError(StatisticalError):
    """Raised when sample ratio mismatch is detected."""

    def __init__(
        self,
        message: str,
        expected_ratio: float,
        actual_ratio: float,
        p_value: float,
        **kwargs: Any,
    ) -> None:
        """Initialize with SRM details."""
        details = {
            "expected_ratio": expected_ratio,
            "actual_ratio": actual_ratio,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
        }
        super().__init__(message, details=details, **kwargs)


class ConvergenceError(StatisticalError):
    """Raised when a statistical algorithm fails to converge."""

    def __init__(
        self,
        message: str,
        iterations: int,
        max_iterations: int,
        tolerance: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with convergence information."""
        details = {
            "iterations": iterations,
            "max_iterations": max_iterations,
        }
        if tolerance:
            details["tolerance"] = tolerance
        super().__init__(message, details=details, **kwargs)


class MultipleTestingError(StatisticalError):
    """Raised when multiple testing correction fails."""

    pass


# ============== Model Exceptions ==============


class ModelError(PortfolioBaseException):
    """Base class for model-related errors."""

    pass


class ModelNotFittedError(ModelError):
    """Raised when trying to use an unfitted model."""

    pass


class ModelValidationError(ModelError):
    """Raised when model validation fails."""

    pass


class FeatureEngineeringError(ModelError):
    """Raised when feature engineering fails."""

    pass


class PredictionError(ModelError):
    """Raised when prediction fails."""

    pass


class CalibrationError(ModelError):
    """Raised when model calibration fails."""

    def __init__(
        self,
        message: str,
        calibration_score: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with calibration details."""
        details = {}
        if calibration_score is not None:
            details["calibration_score"] = calibration_score
        if threshold is not None:
            details["threshold"] = threshold
        super().__init__(message, details=details, **kwargs)


# ============== Experiment Exceptions ==============


class ExperimentError(PortfolioBaseException):
    """Base class for experiment-related errors."""

    pass


class RandomizationError(ExperimentError):
    """Raised when randomization issues are detected."""

    pass


class ExperimentConfigError(ExperimentError):
    """Raised when experiment configuration is invalid."""

    pass


class MetricCalculationError(ExperimentError):
    """Raised when metric calculation fails."""

    pass


class SegmentationError(ExperimentError):
    """Raised when segmentation fails."""

    pass


# ============== Visualization Exceptions ==============


class VisualizationError(PortfolioBaseException):
    """Base class for visualization errors."""

    pass


class PlottingError(VisualizationError):
    """Raised when plotting fails."""

    pass


class InvalidDimensionsError(VisualizationError):
    """Raised when data dimensions are invalid for visualization."""

    def __init__(
        self,
        message: str,
        expected_dims: int,
        actual_dims: int,
        **kwargs: Any,
    ) -> None:
        """Initialize with dimension information."""
        details = {
            "expected_dimensions": expected_dims,
            "actual_dimensions": actual_dims,
        }
        super().__init__(message, details=details, **kwargs)


# ============== IO Exceptions ==============


class IOError(PortfolioBaseException):
    """Base class for input/output errors."""

    pass


class FileNotFoundError(IOError):
    """
    Raised when a required file is missing from disk or cloud storage.

    Attributes
    ----------
    file_path : str
        Absolute or relative path that failed to resolve. Also exposed via the
        ``details`` dict on :class:`PortfolioBaseException`.

    Examples
    --------
    >>> raise FileNotFoundError("Training data missing", file_path="data/churn.csv")
    Traceback (most recent call last):
        ...
    FileNotFoundError: Training data missing
    """

    def __init__(
        self,
        message: str,
        file_path: str,
        **kwargs: Any,
    ) -> None:
        """Attach the offending ``file_path`` to the exception metadata."""
        details = {"file_path": file_path}
        super().__init__(message, details=details, **kwargs)


class InvalidFileFormatError(IOError):
    """Raised when file format is invalid."""

    def __init__(
        self,
        message: str,
        expected_format: str,
        actual_format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with format information."""
        details = {"expected_format": expected_format}
        if actual_format:
            details["actual_format"] = actual_format
        super().__init__(message, details=details, **kwargs)


class DataExportError(IOError):
    """Raised when data export fails."""

    pass


# ============== Configuration Exceptions ==============


class ConfigurationError(PortfolioBaseException):
    """Base class for configuration errors."""

    pass


class InvalidParameterError(ConfigurationError):
    """Raised when a parameter value is invalid."""

    def __init__(
        self,
        message: str,
        parameter: str,
        value: Any,
        valid_range: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with parameter details."""
        details = {
            "parameter": parameter,
            "invalid_value": value,
        }
        if valid_range:
            details["valid_range"] = valid_range
        super().__init__(message, details=details, **kwargs)


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    pass


# ============== Performance Exceptions ==============


class PerformanceError(PortfolioBaseException):
    """Base class for performance-related errors."""

    pass


class MemoryError(PerformanceError):
    """Raised when memory limits are exceeded."""

    def __init__(
        self,
        message: str,
        required_mb: float,
        available_mb: float,
        **kwargs: Any,
    ) -> None:
        """Initialize with memory details."""
        details = {
            "required_mb": required_mb,
            "available_mb": available_mb,
            "shortage_mb": required_mb - available_mb,
        }
        super().__init__(message, details=details, **kwargs)


class TimeoutError(PerformanceError):
    """Raised when operation times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        elapsed_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with timeout details."""
        details = {"timeout_seconds": timeout_seconds}
        if elapsed_seconds:
            details["elapsed_seconds"] = elapsed_seconds
        super().__init__(message, details=details, **kwargs)


# ============== Validation Utilities ==============


def validate_parameter(
    parameter_name: str,
    value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allowed_values: Optional[list] = None,
    parameter_type: Optional[type] = None,
) -> None:
    """
    Validate a parameter value against constraints.

    Args:
        parameter_name: Name of the parameter
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        allowed_values: List of allowed values
        parameter_type: Expected type

    Raises:
        InvalidParameterError: If validation fails
    """
    # Type check
    if parameter_type is not None and not isinstance(value, parameter_type):
        raise InvalidParameterError(
            f"Parameter '{parameter_name}' must be of type {parameter_type.__name__}",
            parameter=parameter_name,
            value=value,
            valid_range=f"Type: {parameter_type.__name__}",
        )

    # Range check
    if min_value is not None and value < min_value:
        raise InvalidParameterError(
            f"Parameter '{parameter_name}' must be >= {min_value}",
            parameter=parameter_name,
            value=value,
            valid_range=f">= {min_value}",
        )

    if max_value is not None and value > max_value:
        raise InvalidParameterError(
            f"Parameter '{parameter_name}' must be <= {max_value}",
            parameter=parameter_name,
            value=value,
            valid_range=f"<= {max_value}",
        )

    # Allowed values check
    if allowed_values is not None and value not in allowed_values:
        raise InvalidParameterError(
            f"Parameter '{parameter_name}' must be one of {allowed_values}",
            parameter=parameter_name,
            value=value,
            valid_range=f"One of: {allowed_values}",
        )


def validate_dataframe(
    df: Any,
    required_columns: Optional[list] = None,
    min_rows: Optional[int] = None,
    max_missing_rate: float = 0.05,
) -> None:
    """
    Validate a pandas DataFrame.

    Args:
        df: Object to validate as DataFrame
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        max_missing_rate: Maximum allowed missing data rate

    Raises:
        DataValidationError: If validation fails
    """
    import pandas as pd

    # Check if it's a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            "Input must be a pandas DataFrame",
            details={"actual_type": type(df).__name__},
        )

    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise MissingDataError(
                f"Required columns missing: {missing_columns}",
                details={
                    "required_columns": required_columns,
                    "actual_columns": list(df.columns),
                    "missing_columns": list(missing_columns),
                },
            )

    # Check minimum rows
    if min_rows and len(df) < min_rows:
        raise InsufficientDataError(
            f"DataFrame has insufficient rows",
            required=min_rows,
            actual=len(df),
        )

    # Check missing data rate
    missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_rate > max_missing_rate:
        raise DataQualityError(
            f"Missing data rate ({missing_rate:.2%}) exceeds threshold ({max_missing_rate:.2%})",
            details={
                "missing_rate": missing_rate,
                "threshold": max_missing_rate,
                "total_missing": df.isnull().sum().sum(),
            },
        )
