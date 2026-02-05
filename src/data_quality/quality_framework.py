"""Data Quality Framework
Comprehensive data validation, quality checks, and monitoring
"""

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import jsonschema
import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from scipy import stats
from scipy.stats import kstest, normaltest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Statistical libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """Data quality severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationStatus(Enum):
    """Validation status"""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class DataQualityRule:
    """Individual data quality rule"""

    rule_id: str
    name: str
    description: str
    rule_type: str  # schema, statistical, business, referential
    severity: DataQualityLevel
    check_function: Callable
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of a validation check"""

    rule_id: str
    rule_name: str
    status: ValidationStatus
    message: str
    severity: DataQualityLevel
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    affected_columns: list[str] = field(default_factory=list)
    affected_rows: list[int] | None = None


@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""

    dataset_name: str
    timestamp: datetime
    total_records: int
    total_columns: int
    validation_results: list[ValidationResult]
    quality_score: float
    summary: dict[str, Any]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class SchemaValidator:
    """Schema validation for structured data"""

    def __init__(self):
        self.schemas = {}
        self.pandera_schemas = {}

    def register_schema(self, schema_name: str, schema: dict | pa.DataFrameSchema):
        """Register a schema for validation"""
        if isinstance(schema, dict):
            self.schemas[schema_name] = schema
        elif isinstance(schema, pa.DataFrameSchema):
            self.pandera_schemas[schema_name] = schema
        else:
            raise ValueError("Schema must be dict or pandera.DataFrameSchema")

    def validate_json_schema(self, data: dict, schema_name: str) -> ValidationResult:
        """Validate JSON data against schema"""
        if schema_name not in self.schemas:
            return ValidationResult(
                rule_id="schema_json",
                rule_name="JSON Schema Validation",
                status=ValidationStatus.SKIPPED,
                message=f"Schema {schema_name} not found",
                severity=DataQualityLevel.HIGH,
            )

        try:
            jsonschema.validate(data, self.schemas[schema_name])
            return ValidationResult(
                rule_id="schema_json",
                rule_name="JSON Schema Validation",
                status=ValidationStatus.PASSED,
                message="Data conforms to JSON schema",
                severity=DataQualityLevel.HIGH,
            )
        except jsonschema.ValidationError as e:
            return ValidationResult(
                rule_id="schema_json",
                rule_name="JSON Schema Validation",
                status=ValidationStatus.FAILED,
                message=f"Schema validation failed: {str(e)}",
                severity=DataQualityLevel.HIGH,
                details={"error": str(e), "path": list(e.path)},
            )

    def validate_dataframe_schema(
        self, df: pd.DataFrame, schema_name: str
    ) -> ValidationResult:
        """Validate DataFrame against pandera schema"""
        if schema_name not in self.pandera_schemas:
            # Try to infer schema
            inferred_schema = self._infer_schema(df)
            self.pandera_schemas[schema_name] = inferred_schema

        schema = self.pandera_schemas[schema_name]

        try:
            schema.validate(df)
            return ValidationResult(
                rule_id="schema_df",
                rule_name="DataFrame Schema Validation",
                status=ValidationStatus.PASSED,
                message="DataFrame conforms to schema",
                severity=DataQualityLevel.HIGH,
            )
        except pa.errors.SchemaError as e:
            return ValidationResult(
                rule_id="schema_df",
                rule_name="DataFrame Schema Validation",
                status=ValidationStatus.FAILED,
                message=f"Schema validation failed: {str(e)}",
                severity=DataQualityLevel.HIGH,
                details={"errors": str(e)},
            )

    def _infer_schema(self, df: pd.DataFrame) -> pa.DataFrameSchema:
        """Infer pandera schema from DataFrame"""
        columns = {}
        for col in df.columns:
            dtype = df[col].dtype
            nullable = df[col].isnull().any()

            if pd.api.types.is_numeric_dtype(dtype):
                checks = [
                    Check.greater_than_or_equal_to(df[col].min()),
                    Check.less_than_or_equal_to(df[col].max()),
                ]
            else:
                checks = []

            columns[col] = Column(dtype, nullable=nullable, checks=checks)

        return DataFrameSchema(columns)


class StatisticalValidator:
    """Statistical distribution and anomaly checks"""

    def __init__(self):
        self.distribution_tests = {
            "normal": normaltest,
            "uniform": lambda x: kstest(x, "uniform"),
            "exponential": lambda x: kstest(x, "expon"),
        }

    def check_distribution(
        self, data: pd.Series, expected_dist: str = "normal", alpha: float = 0.05
    ) -> ValidationResult:
        """Check if data follows expected distribution"""
        if expected_dist not in self.distribution_tests:
            return ValidationResult(
                rule_id="dist_check",
                rule_name=f"Distribution Check ({expected_dist})",
                status=ValidationStatus.SKIPPED,
                message=f"Unknown distribution: {expected_dist}",
                severity=DataQualityLevel.MEDIUM,
            )

        test_func = self.distribution_tests[expected_dist]

        try:
            # Remove NaN values
            clean_data = data.dropna()

            if len(clean_data) < 20:
                return ValidationResult(
                    rule_id="dist_check",
                    rule_name=f"Distribution Check ({expected_dist})",
                    status=ValidationStatus.WARNING,
                    message="Insufficient data for distribution test",
                    severity=DataQualityLevel.LOW,
                    details={"sample_size": len(clean_data)},
                )

            statistic, p_value = test_func(clean_data)[:2]

            if p_value > alpha:
                status = ValidationStatus.PASSED
                message = f"Data follows {expected_dist} distribution (p={p_value:.4f})"
            else:
                status = ValidationStatus.WARNING
                message = f"Data may not follow {expected_dist} distribution (p={p_value:.4f})"

            return ValidationResult(
                rule_id="dist_check",
                rule_name=f"Distribution Check ({expected_dist})",
                status=status,
                message=message,
                severity=DataQualityLevel.MEDIUM,
                details={
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "alpha": alpha,
                },
            )
        except Exception as e:
            return ValidationResult(
                rule_id="dist_check",
                rule_name=f"Distribution Check ({expected_dist})",
                status=ValidationStatus.FAILED,
                message=f"Distribution test failed: {str(e)}",
                severity=DataQualityLevel.MEDIUM,
            )

    def detect_outliers(
        self, df: pd.DataFrame, method: str = "iqr", contamination: float = 0.1
    ) -> ValidationResult:
        """Detect outliers in data"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for col in numeric_cols:
            data = df[col].dropna()

            if method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_mask = (data < lower) | (data > upper)

            elif method == "zscore":
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = z_scores > 3

            elif method == "isolation_forest":
                clf = IsolationForest(contamination=contamination, random_state=42)
                outlier_mask = clf.fit_predict(data.values.reshape(-1, 1)) == -1

            elif method == "lof":
                clf = LocalOutlierFactor(contamination=contamination)
                outlier_mask = clf.fit_predict(data.values.reshape(-1, 1)) == -1

            else:
                continue

            outlier_indices = df.index[outlier_mask].tolist()
            if outlier_indices:
                outliers[col] = {
                    "count": len(outlier_indices),
                    "percentage": len(outlier_indices) / len(data) * 100,
                    "indices": outlier_indices[:100],  # Limit to first 100
                }

        if outliers:
            total_outliers = sum(v["count"] for v in outliers.values())
            status = (
                ValidationStatus.WARNING
                if total_outliers / len(df) < 0.05
                else ValidationStatus.FAILED
            )

            return ValidationResult(
                rule_id="outlier_detection",
                rule_name=f"Outlier Detection ({method})",
                status=status,
                message=f"Found {total_outliers} outliers across {len(outliers)} columns",
                severity=DataQualityLevel.MEDIUM,
                details=outliers,
                affected_columns=list(outliers.keys()),
            )
        else:
            return ValidationResult(
                rule_id="outlier_detection",
                rule_name=f"Outlier Detection ({method})",
                status=ValidationStatus.PASSED,
                message="No outliers detected",
                severity=DataQualityLevel.MEDIUM,
            )

    def check_correlation(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> ValidationResult:
        """Check for highly correlated features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return ValidationResult(
                rule_id="correlation_check",
                rule_name="High Correlation Check",
                status=ValidationStatus.SKIPPED,
                message="Insufficient numeric columns for correlation check",
                severity=DataQualityLevel.LOW,
            )

        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_matrix.iloc[i, j],
                        }
                    )

        if high_corr_pairs:
            return ValidationResult(
                rule_id="correlation_check",
                rule_name="High Correlation Check",
                status=ValidationStatus.WARNING,
                message=f"Found {len(high_corr_pairs)} highly correlated feature pairs",
                severity=DataQualityLevel.MEDIUM,
                details={"high_correlations": high_corr_pairs},
            )
        else:
            return ValidationResult(
                rule_id="correlation_check",
                rule_name="High Correlation Check",
                status=ValidationStatus.PASSED,
                message="No highly correlated features found",
                severity=DataQualityLevel.MEDIUM,
            )

    def check_multicollinearity(
        self, df: pd.DataFrame, threshold: float = 10.0
    ) -> ValidationResult:
        """Check for multicollinearity using VIF"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return ValidationResult(
                rule_id="vif_check",
                rule_name="Multicollinearity Check (VIF)",
                status=ValidationStatus.SKIPPED,
                message="Insufficient numeric columns for VIF check",
                severity=DataQualityLevel.MEDIUM,
            )

        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["feature"] = numeric_cols

        try:
            vif_values = []
            for i in range(len(numeric_cols)):
                vif_values.append(
                    variance_inflation_factor(df[numeric_cols].dropna().values, i)
                )
            vif_data["VIF"] = vif_values

            high_vif = vif_data[vif_data["VIF"] > threshold]

            if not high_vif.empty:
                return ValidationResult(
                    rule_id="vif_check",
                    rule_name="Multicollinearity Check (VIF)",
                    status=ValidationStatus.WARNING,
                    message=f"Found {len(high_vif)} features with high VIF",
                    severity=DataQualityLevel.MEDIUM,
                    details={"high_vif_features": high_vif.to_dict("records")},
                )
            else:
                return ValidationResult(
                    rule_id="vif_check",
                    rule_name="Multicollinearity Check (VIF)",
                    status=ValidationStatus.PASSED,
                    message="No multicollinearity detected",
                    severity=DataQualityLevel.MEDIUM,
                )
        except Exception as e:
            return ValidationResult(
                rule_id="vif_check",
                rule_name="Multicollinearity Check (VIF)",
                status=ValidationStatus.FAILED,
                message=f"VIF calculation failed: {str(e)}",
                severity=DataQualityLevel.MEDIUM,
            )


class DataIntegrityValidator:
    """Referential integrity and business rule validation"""

    def __init__(self):
        self.business_rules = {}
        self.referential_rules = {}

    def register_business_rule(self, rule: DataQualityRule):
        """Register a business rule"""
        self.business_rules[rule.rule_id] = rule

    def register_referential_rule(self, rule: DataQualityRule):
        """Register a referential integrity rule"""
        self.referential_rules[rule.rule_id] = rule

    def check_uniqueness(
        self, df: pd.DataFrame, columns: list[str]
    ) -> ValidationResult:
        """Check uniqueness constraint"""
        duplicates = df.duplicated(subset=columns, keep=False)
        duplicate_count = duplicates.sum()

        if duplicate_count > 0:
            duplicate_rows = df[duplicates].index.tolist()[:100]  # Limit to first 100

            return ValidationResult(
                rule_id="uniqueness_check",
                rule_name="Uniqueness Constraint",
                status=ValidationStatus.FAILED,
                message=f"Found {duplicate_count} duplicate records",
                severity=DataQualityLevel.HIGH,
                details={
                    "duplicate_count": int(duplicate_count),
                    "duplicate_rows": duplicate_rows,
                    "columns_checked": columns,
                },
                affected_columns=columns,
                affected_rows=duplicate_rows,
            )
        else:
            return ValidationResult(
                rule_id="uniqueness_check",
                rule_name="Uniqueness Constraint",
                status=ValidationStatus.PASSED,
                message="All records are unique",
                severity=DataQualityLevel.HIGH,
            )

    def check_referential_integrity(
        self, df1: pd.DataFrame, df2: pd.DataFrame, key1: str, key2: str
    ) -> ValidationResult:
        """Check referential integrity between two dataframes"""
        missing_refs = df1[~df1[key1].isin(df2[key2])]
        missing_count = len(missing_refs)

        if missing_count > 0:
            missing_values = missing_refs[key1].unique()[:20]  # Limit to first 20

            return ValidationResult(
                rule_id="referential_integrity",
                rule_name="Referential Integrity Check",
                status=ValidationStatus.FAILED,
                message=f"Found {missing_count} records with missing references",
                severity=DataQualityLevel.CRITICAL,
                details={
                    "missing_count": missing_count,
                    "missing_values": missing_values.tolist(),
                    "key_column": key1,
                },
                affected_columns=[key1],
            )
        else:
            return ValidationResult(
                rule_id="referential_integrity",
                rule_name="Referential Integrity Check",
                status=ValidationStatus.PASSED,
                message="All references are valid",
                severity=DataQualityLevel.CRITICAL,
            )

    def validate_business_rules(self, df: pd.DataFrame) -> list[ValidationResult]:
        """Validate all registered business rules"""
        results = []

        for rule_id, rule in self.business_rules.items():
            if not rule.enabled:
                continue

            try:
                result = rule.check_function(df, **rule.parameters)
                if isinstance(result, ValidationResult):
                    results.append(result)
                elif isinstance(result, bool):
                    results.append(
                        ValidationResult(
                            rule_id=rule.rule_id,
                            rule_name=rule.name,
                            status=ValidationStatus.PASSED
                            if result
                            else ValidationStatus.FAILED,
                            message=rule.description,
                            severity=rule.severity,
                        )
                    )
            except Exception as e:
                results.append(
                    ValidationResult(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        status=ValidationStatus.FAILED,
                        message=f"Rule execution failed: {str(e)}",
                        severity=rule.severity,
                    )
                )

        return results

    def check_date_consistency(
        self, df: pd.DataFrame, start_col: str, end_col: str
    ) -> ValidationResult:
        """Check if start dates are before end dates"""
        df[start_col] = pd.to_datetime(df[start_col])
        df[end_col] = pd.to_datetime(df[end_col])

        invalid_dates = df[df[start_col] > df[end_col]]
        invalid_count = len(invalid_dates)

        if invalid_count > 0:
            return ValidationResult(
                rule_id="date_consistency",
                rule_name="Date Consistency Check",
                status=ValidationStatus.FAILED,
                message=f"Found {invalid_count} records with invalid date ranges",
                severity=DataQualityLevel.HIGH,
                details={
                    "invalid_count": invalid_count,
                    "invalid_rows": invalid_dates.index.tolist()[:100],
                },
                affected_columns=[start_col, end_col],
            )
        else:
            return ValidationResult(
                rule_id="date_consistency",
                rule_name="Date Consistency Check",
                status=ValidationStatus.PASSED,
                message="All date ranges are valid",
                severity=DataQualityLevel.HIGH,
            )


class DataQualityFramework:
    """Main data quality framework orchestrator"""

    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.statistical_validator = StatisticalValidator()
        self.integrity_validator = DataIntegrityValidator()
        self.validation_history = []
        self.rules = []

    def register_rule(self, rule: DataQualityRule):
        """Register a validation rule"""
        self.rules.append(rule)

    def validate_completeness(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> ValidationResult:
        """Check data completeness"""
        missing_data = df.isnull().sum()
        missing_pct = missing_data / len(df)

        columns_with_missing = missing_pct[missing_pct > (1 - threshold)]

        if not columns_with_missing.empty:
            return ValidationResult(
                rule_id="completeness_check",
                rule_name="Data Completeness Check",
                status=ValidationStatus.WARNING
                if columns_with_missing.max() < 0.5
                else ValidationStatus.FAILED,
                message=f"Found {len(columns_with_missing)} columns below completeness threshold",
                severity=DataQualityLevel.HIGH,
                details={
                    "missing_columns": columns_with_missing.to_dict(),
                    "threshold": threshold,
                },
                affected_columns=columns_with_missing.index.tolist(),
            )
        else:
            return ValidationResult(
                rule_id="completeness_check",
                rule_name="Data Completeness Check",
                status=ValidationStatus.PASSED,
                message="All columns meet completeness threshold",
                severity=DataQualityLevel.HIGH,
            )

    def validate_data_types(
        self, df: pd.DataFrame, expected_types: dict[str, type]
    ) -> ValidationResult:
        """Validate data types"""
        type_mismatches = []

        for col, expected_type in expected_types.items():
            if col not in df.columns:
                type_mismatches.append({"column": col, "issue": "Column not found"})
            elif not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                type_mismatches.append(
                    {
                        "column": col,
                        "expected": str(expected_type),
                        "actual": str(df[col].dtype),
                    }
                )

        if type_mismatches:
            return ValidationResult(
                rule_id="type_check",
                rule_name="Data Type Validation",
                status=ValidationStatus.FAILED,
                message=f"Found {len(type_mismatches)} type mismatches",
                severity=DataQualityLevel.HIGH,
                details={"mismatches": type_mismatches},
            )
        else:
            return ValidationResult(
                rule_id="type_check",
                rule_name="Data Type Validation",
                status=ValidationStatus.PASSED,
                message="All data types are correct",
                severity=DataQualityLevel.HIGH,
            )

    def validate_range(
        self,
        df: pd.DataFrame,
        column: str,
        min_val: float = None,
        max_val: float = None,
    ) -> ValidationResult:
        """Validate value ranges"""
        if column not in df.columns:
            return ValidationResult(
                rule_id="range_check",
                rule_name=f"Range Validation ({column})",
                status=ValidationStatus.FAILED,
                message=f"Column {column} not found",
                severity=DataQualityLevel.HIGH,
            )

        values = df[column].dropna()
        out_of_range = []

        if min_val is not None:
            below_min = values < min_val
            if below_min.any():
                out_of_range.append(f"{below_min.sum()} values below {min_val}")

        if max_val is not None:
            above_max = values > max_val
            if above_max.any():
                out_of_range.append(f"{above_max.sum()} values above {max_val}")

        if out_of_range:
            return ValidationResult(
                rule_id="range_check",
                rule_name=f"Range Validation ({column})",
                status=ValidationStatus.FAILED,
                message=f"Values out of range: {', '.join(out_of_range)}",
                severity=DataQualityLevel.HIGH,
                details={
                    "min_val": min_val,
                    "max_val": max_val,
                    "actual_min": float(values.min()),
                    "actual_max": float(values.max()),
                },
                affected_columns=[column],
            )
        else:
            return ValidationResult(
                rule_id="range_check",
                rule_name=f"Range Validation ({column})",
                status=ValidationStatus.PASSED,
                message="All values within expected range",
                severity=DataQualityLevel.HIGH,
            )

    def run_validation(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        rules: list[DataQualityRule] | None = None,
    ) -> DataQualityReport:
        """Run comprehensive data validation"""
        validation_results = []
        timestamp = datetime.now()

        # Basic data checks
        validation_results.append(self.validate_completeness(df))

        # Statistical checks
        for col in df.select_dtypes(include=[np.number]).columns:
            validation_results.append(
                self.statistical_validator.check_distribution(df[col], "normal")
            )

        # Outlier detection
        validation_results.append(
            self.statistical_validator.detect_outliers(df, method="iqr")
        )

        # Correlation checks
        validation_results.append(self.statistical_validator.check_correlation(df))

        # Custom rules
        if rules:
            for rule in rules:
                try:
                    result = rule.check_function(df, **rule.parameters)
                    validation_results.append(result)
                except Exception as e:
                    validation_results.append(
                        ValidationResult(
                            rule_id=rule.rule_id,
                            rule_name=rule.name,
                            status=ValidationStatus.FAILED,
                            message=f"Rule execution failed: {str(e)}",
                            severity=rule.severity,
                        )
                    )

        # Calculate quality score
        passed = sum(
            1 for r in validation_results if r.status == ValidationStatus.PASSED
        )
        total = len(validation_results)
        quality_score = (passed / total * 100) if total > 0 else 0

        # Generate summary
        summary = {
            "total_checks": total,
            "passed": passed,
            "failed": sum(
                1 for r in validation_results if r.status == ValidationStatus.FAILED
            ),
            "warnings": sum(
                1 for r in validation_results if r.status == ValidationStatus.WARNING
            ),
            "critical_issues": sum(
                1
                for r in validation_results
                if r.status == ValidationStatus.FAILED
                and r.severity == DataQualityLevel.CRITICAL
            ),
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results)

        report = DataQualityReport(
            dataset_name=dataset_name,
            timestamp=timestamp,
            total_records=len(df),
            total_columns=len(df.columns),
            validation_results=validation_results,
            quality_score=quality_score,
            summary=summary,
            recommendations=recommendations,
            metadata={
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
            },
        )

        # Store in history
        self.validation_history.append(report)

        return report

    def _generate_recommendations(self, results: list[ValidationResult]) -> list[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        for result in results:
            if result.status == ValidationStatus.FAILED:
                if result.rule_id == "completeness_check":
                    recommendations.append(
                        "Consider imputation strategies for missing data"
                    )
                elif result.rule_id == "outlier_detection":
                    recommendations.append("Review and handle detected outliers")
                elif result.rule_id == "correlation_check":
                    recommendations.append(
                        "Consider removing highly correlated features"
                    )
                elif result.rule_id == "uniqueness_check":
                    recommendations.append("Remove or consolidate duplicate records")

            elif result.status == ValidationStatus.WARNING:
                if result.rule_id == "dist_check":
                    recommendations.append(
                        "Consider data transformation for non-normal distributions"
                    )

        return list(set(recommendations))  # Remove duplicates

    def export_report(self, report: DataQualityReport, format: str = "json") -> str:
        """Export quality report in various formats"""
        if format == "json":
            return json.dumps(asdict(report), default=str, indent=2)
        elif format == "html":
            return self._generate_html_report(report)
        elif format == "markdown":
            return self._generate_markdown_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_html_report(self, report: DataQualityReport) -> str:
        """Generate HTML report"""
        html = f"""
        <html>
        <head>
            <title>Data Quality Report - {report.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; }}
                .score {{ font-size: 48px; font-weight: bold;
                         color: {"green" if report.quality_score > 80 else "orange" if report.quality_score > 60 else "red"}; }}
                .summary {{ margin: 20px 0; }}
                .results {{ margin: 20px 0; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .warning {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report: {report.dataset_name}</h1>
                <p>Generated: {report.timestamp}</p>
                <div class="score">Quality Score: {report.quality_score:.1f}%</div>
            </div>

            <div class="summary">
                <h2>Summary</h2>
                <ul>
                    <li>Total Records: {report.total_records:,}</li>
                    <li>Total Columns: {report.total_columns}</li>
                    <li>Checks Passed: {report.summary["passed"]}/{report.summary["total_checks"]}</li>
                    <li>Critical Issues: {report.summary["critical_issues"]}</li>
                </ul>
            </div>

            <div class="results">
                <h2>Validation Results</h2>
                <table>
                    <tr>
                        <th>Check</th>
                        <th>Status</th>
                        <th>Severity</th>
                        <th>Message</th>
                    </tr>
        """

        for result in report.validation_results:
            status_class = result.status.value
            html += f"""
                    <tr>
                        <td>{result.rule_name}</td>
                        <td class="{status_class}">{result.status.value.upper()}</td>
                        <td>{result.severity.value}</td>
                        <td>{result.message}</td>
                    </tr>
            """

        html += """
                </table>
            </div>

            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
        """

        for rec in report.recommendations:
            html += f"<li>{rec}</li>"

        html += """
                </ul>
            </div>
        </body>
        </html>
        """

        return html

    def _generate_markdown_report(self, report: DataQualityReport) -> str:
        """Generate Markdown report"""
        md = f"""# Data Quality Report: {report.dataset_name}

**Generated:** {report.timestamp}

## Quality Score: {report.quality_score:.1f}%

## Summary
- **Total Records:** {report.total_records:,}
- **Total Columns:** {report.total_columns}
- **Checks Passed:** {report.summary["passed"]}/{report.summary["total_checks"]}
- **Failed:** {report.summary["failed"]}
- **Warnings:** {report.summary["warnings"]}
- **Critical Issues:** {report.summary["critical_issues"]}

## Validation Results

| Check | Status | Severity | Message |
|-------|--------|----------|---------|
"""

        for result in report.validation_results:
            md += f"| {result.rule_name} | {result.status.value.upper()} | {result.severity.value} | {result.message} |\n"

        md += "\n## Recommendations\n\n"
        for rec in report.recommendations:
            md += f"- {rec}\n"

        return md


# Example usage
if __name__ == "__main__":
    # Create sample data
    df = pd.DataFrame(
        {
            "id": range(1000),
            "age": np.random.normal(35, 10, 1000),
            "salary": np.random.normal(50000, 15000, 1000),
            "department": np.random.choice(["IT", "HR", "Sales"], 1000),
            "start_date": pd.date_range("2020-01-01", periods=1000, freq="D"),
            "end_date": pd.date_range("2021-01-01", periods=1000, freq="D"),
        }
    )

    # Add some data quality issues
    df.loc[50:100, "age"] = np.nan
    df.loc[200:210, "salary"] = -1000
    df.loc[300:305, "id"] = 1  # Duplicates

    # Initialize framework
    dq_framework = DataQualityFramework()

    # Run validation
    report = dq_framework.run_validation(df, dataset_name="employee_data")

    # Print report
    print(f"Quality Score: {report.quality_score:.1f}%")
    print(f"Summary: {report.summary}")
    print(f"Recommendations: {report.recommendations}")

    # Export report
    html_report = dq_framework.export_report(report, format="html")
    with open("data_quality_report.html", "w") as f:
        f.write(html_report)
