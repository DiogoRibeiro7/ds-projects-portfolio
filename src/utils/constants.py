"""Constants and configuration values for the data science portfolio.

This module centralizes all magic numbers and configuration constants
to improve code maintainability and readability.
"""

from typing import Final

# ============== Statistical Constants ==============

# Significance levels
DEFAULT_ALPHA: Final[float] = 0.05
STRICT_ALPHA: Final[float] = 0.01
EXPLORATORY_ALPHA: Final[float] = 0.10

# Statistical power
DEFAULT_POWER: Final[float] = 0.80
HIGH_POWER: Final[float] = 0.90
MIN_ACCEPTABLE_POWER: Final[float] = 0.70

# Sample size thresholds
MIN_SAMPLE_SIZE: Final[int] = 30
MIN_GROUP_SIZE: Final[int] = 100
LARGE_SAMPLE_THRESHOLD: Final[int] = 1000

# Effect size thresholds (Cohen's d)
SMALL_EFFECT_SIZE: Final[float] = 0.20
MEDIUM_EFFECT_SIZE: Final[float] = 0.50
LARGE_EFFECT_SIZE: Final[float] = 0.80
MIN_DETECTABLE_EFFECT: Final[float] = 0.10

# Bootstrap parameters
DEFAULT_BOOTSTRAP_ITERATIONS: Final[int] = 5000
MIN_BOOTSTRAP_ITERATIONS: Final[int] = 1000
MAX_BOOTSTRAP_ITERATIONS: Final[int] = 10000

# ============== Data Quality Constants ==============

# Missing data thresholds
MAX_MISSING_RATE: Final[float] = 0.05
CRITICAL_MISSING_RATE: Final[float] = 0.20
WARNING_MISSING_RATE: Final[float] = 0.10

# Outlier detection parameters
IQR_MULTIPLIER: Final[float] = 1.5
ZSCORE_THRESHOLD: Final[float] = 3.0
ISOLATION_FOREST_CONTAMINATION: Final[float] = 0.10

# Sample ratio thresholds
MIN_SAMPLE_RATIO: Final[float] = 0.80
CRITICAL_SAMPLE_RATIO: Final[float] = 0.50
IDEAL_SAMPLE_RATIO: Final[float] = 1.00

# Data quality scores
EXCELLENT_QUALITY_SCORE: Final[int] = 90
GOOD_QUALITY_SCORE: Final[int] = 80
ACCEPTABLE_QUALITY_SCORE: Final[int] = 70
POOR_QUALITY_SCORE: Final[int] = 60
CRITICAL_QUALITY_SCORE: Final[int] = 50

# ============== Time Constants ==============

# Date formats
DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d"
DATETIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
ISO_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"

# Time windows (in days)
SHORT_WINDOW: Final[int] = 7
MEDIUM_WINDOW: Final[int] = 14
LONG_WINDOW: Final[int] = 30
QUARTERLY_WINDOW: Final[int] = 90

# ============== Machine Learning Constants ==============

# Cross-validation
DEFAULT_CV_FOLDS: Final[int] = 5
STRATIFIED_CV_FOLDS: Final[int] = 10
TIME_SERIES_CV_FOLDS: Final[int] = 3

# Train/test split ratios
DEFAULT_TEST_SIZE: Final[float] = 0.20
VALIDATION_SIZE: Final[float] = 0.15
MIN_TRAIN_SIZE: Final[float] = 0.60

# Model thresholds
DEFAULT_CLASSIFICATION_THRESHOLD: Final[float] = 0.50
HIGH_PRECISION_THRESHOLD: Final[float] = 0.70
HIGH_RECALL_THRESHOLD: Final[float] = 0.30

# Feature engineering
MAX_CORRELATION_THRESHOLD: Final[float] = 0.95
MIN_VARIANCE_THRESHOLD: Final[float] = 0.01
MAX_CARDINALITY_RATIO: Final[float] = 0.95

# ============== Visualization Constants ==============

# Figure sizes (inches)
SMALL_FIGURE_SIZE: Final[tuple] = (8, 6)
MEDIUM_FIGURE_SIZE: Final[tuple] = (10, 8)
LARGE_FIGURE_SIZE: Final[tuple] = (14, 10)
WIDE_FIGURE_SIZE: Final[tuple] = (16, 6)

# Color palettes
DEFAULT_PALETTE: Final[str] = "husl"
CATEGORICAL_PALETTE: Final[str] = "Set2"
SEQUENTIAL_PALETTE: Final[str] = "viridis"
DIVERGING_PALETTE: Final[str] = "coolwarm"

# Plot parameters
DEFAULT_DPI: Final[int] = 100
HIGH_DPI: Final[int] = 300
DEFAULT_PLOT_ALPHA: Final[float] = 0.7
GRID_ALPHA: Final[float] = 0.3

# ============== Performance Constants ==============

# Memory limits (MB)
MAX_MEMORY_MB: Final[int] = 4096
WARNING_MEMORY_MB: Final[int] = 2048
CHUNK_SIZE_MB: Final[int] = 100

# Processing thresholds
MAX_ROWS_IN_MEMORY: Final[int] = 1_000_000
BATCH_SIZE: Final[int] = 10_000
PARALLEL_THRESHOLD: Final[int] = 100_000

# Cache settings
CACHE_TTL_SECONDS: Final[int] = 3600
MAX_CACHE_SIZE_MB: Final[int] = 512

# ============== API and Network Constants ==============

# HTTP settings
DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
MAX_RETRIES: Final[int] = 3
BACKOFF_FACTOR: Final[float] = 1.5

# Rate limiting
MAX_REQUESTS_PER_MINUTE: Final[int] = 60
MAX_REQUESTS_PER_HOUR: Final[int] = 1000
RATE_LIMIT_WINDOW: Final[int] = 60

# ============== File and Path Constants ==============

# File size limits (MB)
MAX_FILE_SIZE_MB: Final[int] = 100
MAX_UPLOAD_SIZE_MB: Final[int] = 50
MAX_CSV_SIZE_MB: Final[int] = 200

# File extensions
ALLOWED_DATA_EXTENSIONS: Final[tuple] = (
    ".csv",
    ".xlsx",
    ".json",
    ".parquet",
    ".feather",
)
ALLOWED_IMAGE_EXTENSIONS: Final[tuple] = (".png", ".jpg", ".jpeg", ".gif", ".svg")
ALLOWED_NOTEBOOK_EXTENSIONS: Final[tuple] = (".ipynb", ".py")

# ============== Logging Constants ==============

# Log levels
DEBUG_LEVEL: Final[str] = "DEBUG"
INFO_LEVEL: Final[str] = "INFO"
WARNING_LEVEL: Final[str] = "WARNING"
ERROR_LEVEL: Final[str] = "ERROR"
CRITICAL_LEVEL: Final[str] = "CRITICAL"

# Log format
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_LOG_FORMAT: Final[str] = "%(levelname)s: %(message)s"

# ============== Business Constants ==============

# Business metrics thresholds
MIN_ROI: Final[float] = 1.0
TARGET_ROI: Final[float] = 3.0
MIN_CONVERSION_RATE: Final[float] = 0.01
TARGET_CONVERSION_RATE: Final[float] = 0.05

# Cost factors
DEFAULT_CUSTOMER_VALUE: Final[float] = 100.0
DEFAULT_ACQUISITION_COST: Final[float] = 20.0
DEFAULT_RETENTION_COST: Final[float] = 5.0

# ============== Experiment Constants ==============

# A/B test parameters
MIN_EXPERIMENT_DURATION_DAYS: Final[int] = 7
RECOMMENDED_DURATION_DAYS: Final[int] = 14
MAX_EXPERIMENT_DURATION_DAYS: Final[int] = 90

# Multiple testing corrections
BONFERRONI: Final[str] = "bonferroni"
HOLM: Final[str] = "holm"
FDR_BH: Final[str] = "fdr_bh"
FDR_BY: Final[str] = "fdr_by"

# Sequential testing
OBRIEN_FLEMING: Final[str] = "obrien_fleming"
POCOCK: Final[str] = "pocock"
HAYBITTLE_PETO: Final[str] = "haybittle_peto"
