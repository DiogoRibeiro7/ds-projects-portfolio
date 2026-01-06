"""
Notebook validation utilities for comprehensive testing of Jupyter notebooks.
"""

import ast
import json
import os
import re
import sys
import tracemalloc
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat
import numpy as np
import pandas as pd
import psutil
from memory_profiler import memory_usage
from nbformat import NotebookNode


class NotebookValidator:
    """Validates Jupyter notebooks for best practices and common issues."""

    def __init__(self, notebook_path: str):
        """Initialize validator with a notebook path."""
        self.notebook_path = Path(notebook_path)
        self.notebook_name = self.notebook_path.name
        self.issues = []
        self.warnings = []
        self.info = []

        with open(notebook_path, "r", encoding="utf-8") as f:
            self.notebook = nbformat.read(f, as_version=4)

    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks on the notebook."""
        results = {
            "notebook": str(self.notebook_path),
            "timestamp": datetime.now().isoformat(),
            "checks": {},
        }

        # Run all validation methods
        results["checks"]["hardcoded_paths"] = self.check_hardcoded_paths()
        results["checks"]["data_files"] = self.check_data_files()
        results["checks"]["random_seeds"] = self.check_random_seeds()
        results["checks"]["visualizations"] = self.check_visualizations()
        results["checks"]["memory_leaks"] = self.check_memory_patterns()
        results["checks"]["code_quality"] = self.check_code_quality()
        results["checks"]["imports"] = self.check_imports()
        results["checks"]["markdown"] = self.check_markdown_quality()

        results["issues"] = self.issues
        results["warnings"] = self.warnings
        results["info"] = self.info
        results["score"] = self._calculate_score()

        return results

    def check_hardcoded_paths(self) -> Dict[str, Any]:
        """Check for hardcoded paths that should be parameterized."""
        hardcoded_paths = []
        path_patterns = [
            r'["\']\/[A-Za-z0-9_\-\/\.]+(\.csv|\.xlsx|\.json|\.parquet|\.pkl|\.txt)["\']',
            r'["\']C:\\[A-Za-z0-9_\-\\\.]+(\.csv|\.xlsx|\.json|\.parquet|\.pkl|\.txt)["\']',
            r'["\'][A-Za-z]:\/[A-Za-z0-9_\-\/\.]+(\.csv|\.xlsx|\.json|\.parquet|\.pkl|\.txt)["\']',
            r'pd\.read_[a-z]+\(["\'][^"\']*\.(csv|xlsx|json|parquet|pkl)["\']',
            r'open\(["\'][^"\']*\.(csv|xlsx|json|parquet|pkl|txt)["\']',
        ]

        for cell_num, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                source = cell.source
                for pattern in path_patterns:
                    matches = re.findall(pattern, source)
                    if matches:
                        for match in matches:
                            # Skip relative paths starting with '.' or containing 'data/'
                            if not any(
                                skip in str(match)
                                for skip in [
                                    "./data",
                                    "../data",
                                    "data/",
                                    '"data/',
                                    "'data/",
                                ]
                            ):
                                hardcoded_paths.append(
                                    {
                                        "cell": cell_num + 1,
                                        "path": match,
                                        "line": (
                                            source[:100] + "..."
                                            if len(source) > 100
                                            else source
                                        ),
                                    }
                                )

        if hardcoded_paths:
            self.issues.append(
                {
                    "type": "hardcoded_paths",
                    "message": f"Found {len(hardcoded_paths)} hardcoded paths",
                    "details": hardcoded_paths,
                }
            )

        return {
            "passed": len(hardcoded_paths) == 0,
            "count": len(hardcoded_paths),
            "details": hardcoded_paths,
        }

    def check_data_files(self) -> Dict[str, Any]:
        """Verify all referenced data files exist."""
        missing_files = []
        found_files = []

        # Patterns to find file references
        file_patterns = [
            r'pd\.read_csv\(["\']([^"\']+)["\']',
            r'pd\.read_excel\(["\']([^"\']+)["\']',
            r'pd\.read_json\(["\']([^"\']+)["\']',
            r'pd\.read_parquet\(["\']([^"\']+)["\']',
            r'open\(["\']([^"\']+)["\']',
            r'np\.load\(["\']([^"\']+)["\']',
            r'pd\.read_pickle\(["\']([^"\']+)["\']',
        ]

        base_dir = self.notebook_path.parent

        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                source = cell.source
                for pattern in file_patterns:
                    matches = re.findall(pattern, source)
                    for match in matches:
                        # Resolve relative paths
                        if not match.startswith("http"):  # Skip URLs
                            file_path = base_dir / match
                            if file_path.exists():
                                found_files.append(str(match))
                            else:
                                # Try from project root
                                project_root = Path.cwd()
                                file_path_root = project_root / match
                                if file_path_root.exists():
                                    found_files.append(str(match))
                                else:
                                    missing_files.append(str(match))

        if missing_files:
            self.warnings.append(
                {
                    "type": "missing_data_files",
                    "message": f"Found {len(missing_files)} missing data files",
                    "details": missing_files,
                }
            )

        return {
            "passed": len(missing_files) == 0,
            "missing_count": len(missing_files),
            "found_count": len(found_files),
            "missing_files": missing_files,
            "found_files": found_files,
        }

    def check_random_seeds(self) -> Dict[str, Any]:
        """Ensure random seeds are set for reproducibility."""
        seed_patterns = {
            "numpy": r"np\.random\.seed\(\d+\)",
            "random": r"random\.seed\(\d+\)",
            "tensorflow": r"tf\.random\.set_seed\(\d+\)",
            "torch": r"torch\.manual_seed\(\d+\)",
            "sklearn": r"random_state\s*=\s*\d+",
        }

        found_seeds = {lib: False for lib in seed_patterns}
        seed_locations = []

        # Check which libraries are imported
        imported_libs = set()
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                source = cell.source
                if "import numpy" in source or "from numpy" in source:
                    imported_libs.add("numpy")
                if "import random" in source or "from random" in source:
                    imported_libs.add("random")
                if "import tensorflow" in source or "from tensorflow" in source:
                    imported_libs.add("tensorflow")
                if "import torch" in source or "from torch" in source:
                    imported_libs.add("torch")
                if "from sklearn" in source or "import sklearn" in source:
                    imported_libs.add("sklearn")

        # Check for seed settings
        for cell_num, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                source = cell.source
                for lib, pattern in seed_patterns.items():
                    if re.search(pattern, source):
                        found_seeds[lib] = True
                        seed_locations.append(
                            {
                                "cell": cell_num + 1,
                                "library": lib,
                                "pattern_found": re.search(pattern, source).group(),
                            }
                        )

        # Check if imported libs have seeds
        missing_seeds = []
        for lib in imported_libs:
            if lib in found_seeds and not found_seeds[lib]:
                missing_seeds.append(lib)

        if missing_seeds:
            self.warnings.append(
                {
                    "type": "missing_random_seeds",
                    "message": f'Missing random seeds for: {", ".join(missing_seeds)}',
                    "details": {"missing": missing_seeds, "found": seed_locations},
                }
            )

        return {
            "passed": len(missing_seeds) == 0,
            "missing_seeds": missing_seeds,
            "found_seeds": seed_locations,
            "imported_libs": list(imported_libs),
        }

    def check_visualizations(self) -> Dict[str, Any]:
        """Check that visualizations are properly configured."""
        viz_patterns = {
            "matplotlib": [r"plt\.show\(\)", r"plt\.savefig\(", r"fig\.show\(\)"],
            "seaborn": [r"sns\.[a-z]+plot\(", r"sns\.set"],
            "plotly": [r"fig\.show\(\)", r"fig\.write_html\(", r"pio\.show\("],
            "altair": [r"\.interactive\(\)", r"\.save\("],
        }

        found_viz = []
        missing_show = []

        for cell_num, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                source = cell.source

                # Check for plotting commands
                has_plot = False
                has_show = False

                for lib, patterns in viz_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, source):
                            if (
                                "show" in pattern
                                or "save" in pattern
                                or "interactive" in pattern
                            ):
                                has_show = True
                            else:
                                has_plot = True

                # Check for common plotting functions without show
                plot_commands = [
                    "plt.plot",
                    "plt.scatter",
                    "plt.bar",
                    "plt.hist",
                    "sns.",
                    "px.",
                    "go.",
                    "alt.Chart",
                ]
                for cmd in plot_commands:
                    if cmd in source:
                        has_plot = True
                        break

                if has_plot and not has_show and "show()" not in source:
                    missing_show.append(
                        {
                            "cell": cell_num + 1,
                            "issue": "Plot created but not displayed",
                        }
                    )

                if has_plot or has_show:
                    found_viz.append(
                        {
                            "cell": cell_num + 1,
                            "has_plot": has_plot,
                            "has_show": has_show,
                        }
                    )

        if missing_show:
            self.warnings.append(
                {
                    "type": "visualization_issues",
                    "message": f"Found {len(missing_show)} plots without show/display",
                    "details": missing_show,
                }
            )

        return {
            "passed": len(missing_show) == 0,
            "visualization_count": len(found_viz),
            "missing_show_count": len(missing_show),
            "details": {"found": found_viz, "issues": missing_show},
        }

    def check_memory_patterns(self) -> Dict[str, Any]:
        """Check for potential memory leak patterns."""
        memory_issues = []

        patterns = {
            "large_loop_append": r"for .+ in .+:\s*\n\s*[a-z_]+\.append\(",
            "global_accumulator": r"^[a-z_]+ = \[\]",
            "no_del_large_var": r"= pd\.read_|= np\.load|= pickle\.load",
            "repeated_read": r"(pd\.read_[a-z]+|np\.load)\([^)]+\)",
            "infinite_loop_risk": r"while True:|while 1:",
        }

        for cell_num, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                source = cell.source

                # Check for memory-intensive patterns
                for issue_type, pattern in patterns.items():
                    if re.search(pattern, source, re.MULTILINE):
                        memory_issues.append(
                            {
                                "cell": cell_num + 1,
                                "issue_type": issue_type,
                                "pattern": pattern,
                                "severity": (
                                    "high" if "infinite" in issue_type else "medium"
                                ),
                            }
                        )

                # Check for large data operations without cleanup
                if "pd.read_" in source or "np.load" in source:
                    if "del " not in source and cell_num < len(self.notebook.cells) - 1:
                        # Check if variable is deleted in next cells
                        var_match = re.search(r"([a-z_]+)\s*=\s*pd\.read_", source)
                        if var_match:
                            var_name = var_match.group(1)
                            has_cleanup = False
                            for next_cell in self.notebook.cells[
                                cell_num + 1 : cell_num + 3
                            ]:
                                if (
                                    next_cell.cell_type == "code"
                                    and f"del {var_name}" in next_cell.source
                                ):
                                    has_cleanup = True
                                    break
                            if not has_cleanup:
                                self.info.append(
                                    {
                                        "type": "potential_memory_issue",
                                        "cell": cell_num + 1,
                                        "message": f"Large data loaded but no cleanup found for variable: {var_name}",
                                    }
                                )

        if memory_issues:
            self.warnings.append(
                {
                    "type": "memory_patterns",
                    "message": f"Found {len(memory_issues)} potential memory issues",
                    "details": memory_issues,
                }
            )

        return {
            "passed": len([i for i in memory_issues if i.get("severity") == "high"])
            == 0,
            "issue_count": len(memory_issues),
            "high_severity": len(
                [i for i in memory_issues if i.get("severity") == "high"]
            ),
            "details": memory_issues,
        }

    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        quality_issues = []

        for cell_num, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                source = cell.source
                lines = source.split("\n")

                # Check line length
                long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 100]
                if long_lines:
                    quality_issues.append(
                        {
                            "cell": cell_num + 1,
                            "issue": "long_lines",
                            "lines": long_lines[:5],  # Show first 5
                        }
                    )

                # Check for commented code
                commented_code = len(
                    [
                        l
                        for l in lines
                        if l.strip().startswith("#") and len(l.strip()) > 2
                    ]
                )
                if commented_code > 10:
                    quality_issues.append(
                        {
                            "cell": cell_num + 1,
                            "issue": "excessive_comments",
                            "count": commented_code,
                        }
                    )

                # Check for print statements (should use logging)
                print_count = source.count("print(")
                if print_count > 5:
                    quality_issues.append(
                        {
                            "cell": cell_num + 1,
                            "issue": "excessive_prints",
                            "count": print_count,
                        }
                    )

                # Check for bare exceptions
                if "except:" in source or "except Exception:" in source:
                    quality_issues.append(
                        {
                            "cell": cell_num + 1,
                            "issue": "bare_exception",
                            "severity": "high",
                        }
                    )

                # Check for unused imports (simple check)
                imports = re.findall(r"import (\w+)", source)
                imports.extend(re.findall(r"from \w+ import (\w+)", source))
                for imp in imports:
                    if source.count(imp) == 1:  # Only appears in import
                        quality_issues.append(
                            {
                                "cell": cell_num + 1,
                                "issue": "potentially_unused_import",
                                "import": imp,
                            }
                        )

        if quality_issues:
            self.info.append(
                {
                    "type": "code_quality",
                    "message": f"Found {len(quality_issues)} code quality issues",
                    "details": quality_issues,
                }
            )

        return {
            "passed": len([i for i in quality_issues if i.get("severity") == "high"])
            == 0,
            "issue_count": len(quality_issues),
            "details": quality_issues,
        }

    def check_imports(self) -> Dict[str, Any]:
        """Check import organization and dependencies."""
        import_issues = []
        all_imports = []

        for cell_num, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                source = cell.source

                # Find all imports
                imports = re.findall(
                    r"^(import .+|from .+ import .+)$", source, re.MULTILINE
                )
                if imports:
                    all_imports.extend([(cell_num + 1, imp) for imp in imports])

                # Check if imports are not at the beginning
                if cell_num > 2 and imports:
                    import_issues.append(
                        {
                            "cell": cell_num + 1,
                            "issue": "late_imports",
                            "imports": imports,
                        }
                    )

        # Check for duplicate imports
        import_statements = [imp[1] for imp in all_imports]
        duplicates = set(
            [x for x in import_statements if import_statements.count(x) > 1]
        )
        if duplicates:
            import_issues.append(
                {"issue": "duplicate_imports", "duplicates": list(duplicates)}
            )

        if import_issues:
            self.warnings.append(
                {
                    "type": "import_organization",
                    "message": f"Found {len(import_issues)} import issues",
                    "details": import_issues,
                }
            )

        return {
            "passed": len(import_issues) == 0,
            "total_imports": len(all_imports),
            "issues": import_issues,
        }

    def check_markdown_quality(self) -> Dict[str, Any]:
        """Check markdown cell quality."""
        markdown_stats = {
            "total_cells": 0,
            "empty_cells": 0,
            "no_headers": True,
            "avg_length": 0,
        }

        markdown_cells = [
            cell for cell in self.notebook.cells if cell.cell_type == "markdown"
        ]
        markdown_stats["total_cells"] = len(markdown_cells)

        if markdown_cells:
            lengths = []
            for cell in markdown_cells:
                source = cell.source.strip()
                if not source:
                    markdown_stats["empty_cells"] += 1
                else:
                    lengths.append(len(source))
                    if source.startswith("#"):
                        markdown_stats["no_headers"] = False

            if lengths:
                markdown_stats["avg_length"] = sum(lengths) / len(lengths)

        # Check if notebook has any documentation
        if markdown_stats["total_cells"] == 0:
            self.warnings.append(
                {
                    "type": "no_documentation",
                    "message": "Notebook has no markdown cells for documentation",
                }
            )
        elif markdown_stats["total_cells"] < 3:
            self.info.append(
                {
                    "type": "minimal_documentation",
                    "message": f'Notebook has only {markdown_stats["total_cells"]} markdown cells',
                }
            )

        return {
            "passed": markdown_stats["total_cells"] > 0
            and markdown_stats["empty_cells"] == 0,
            "stats": markdown_stats,
        }

    def _calculate_score(self) -> float:
        """Calculate overall notebook quality score."""
        total_issues = len(self.issues)
        total_warnings = len(self.warnings)
        total_info = len(self.info)

        # Start with perfect score
        score = 100.0

        # Deduct points for issues
        score -= total_issues * 10  # Major issues
        score -= total_warnings * 5  # Warnings
        score -= total_info * 2  # Info/suggestions

        return max(0, min(100, score))


class DataValidator:
    """Validates data quality in notebooks."""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, name: str = "data") -> Dict[str, Any]:
        """Validate a pandas DataFrame for quality issues."""
        results = {"name": name, "shape": df.shape, "checks": {}}

        # Check for nulls
        null_counts = df.isnull().sum()
        if null_counts.any():
            results["checks"]["nulls"] = {
                "found": True,
                "columns": null_counts[null_counts > 0].to_dict(),
                "total_nulls": null_counts.sum(),
            }
        else:
            results["checks"]["nulls"] = {"found": False}

        # Check for duplicates
        duplicates = df.duplicated().sum()
        results["checks"]["duplicates"] = {
            "count": duplicates,
            "percentage": (duplicates / len(df)) * 100 if len(df) > 0 else 0,
        }

        # Check for outliers (numerical columns)
        outlier_info = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (
                (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            ).sum()
            if outliers > 0:
                outlier_info[col] = {
                    "count": outliers,
                    "percentage": (outliers / len(df)) * 100,
                }

        if outlier_info:
            results["checks"]["outliers"] = outlier_info

        # Check data types
        results["checks"]["dtypes"] = df.dtypes.astype(str).to_dict()

        # Check for high cardinality in categorical columns
        high_cardinality = {}
        for col in df.select_dtypes(include=["object"]).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:  # More than 50% unique values
                high_cardinality[col] = {
                    "unique_values": df[col].nunique(),
                    "unique_ratio": unique_ratio,
                }

        if high_cardinality:
            results["checks"]["high_cardinality"] = high_cardinality

        # Basic statistics
        results["stats"] = {
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "column_count": len(df.columns),
            "row_count": len(df),
        }

        return results

    @staticmethod
    def validate_schema(
        df: pd.DataFrame, expected_schema: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validate DataFrame against expected schema."""
        results = {
            "matches": True,
            "missing_columns": [],
            "extra_columns": [],
            "type_mismatches": {},
        }

        actual_columns = set(df.columns)
        expected_columns = set(expected_schema.keys())

        # Check for missing columns
        missing = expected_columns - actual_columns
        if missing:
            results["matches"] = False
            results["missing_columns"] = list(missing)

        # Check for extra columns
        extra = actual_columns - expected_columns
        if extra:
            results["extra_columns"] = list(extra)

        # Check data types
        for col, expected_type in expected_schema.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not DataValidator._types_compatible(actual_type, expected_type):
                    results["matches"] = False
                    results["type_mismatches"][col] = {
                        "expected": expected_type,
                        "actual": actual_type,
                    }

        return results

    @staticmethod
    def _types_compatible(actual: str, expected: str) -> bool:
        """Check if data types are compatible."""
        type_groups = {
            "numeric": ["int", "float", "number"],
            "string": ["object", "str", "string"],
            "datetime": ["datetime", "timestamp"],
            "boolean": ["bool", "boolean"],
        }

        for group in type_groups.values():
            if any(t in actual.lower() for t in group) and any(
                t in expected.lower() for t in group
            ):
                return True

        return actual.lower() == expected.lower()

    @staticmethod
    def validate_metrics(
        metrics: Dict[str, float], expected_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Validate that metrics are within expected ranges."""
        results = {"all_valid": True, "out_of_range": {}}

        for metric, value in metrics.items():
            if metric in expected_ranges:
                min_val, max_val = expected_ranges[metric]
                if not (min_val <= value <= max_val):
                    results["all_valid"] = False
                    results["out_of_range"][metric] = {
                        "value": value,
                        "expected_range": (min_val, max_val),
                    }

        return results
