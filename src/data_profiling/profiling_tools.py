"""Data Profiling Tools
Automatic data quality reports, lineage tracking, versioning, and catalog integration
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Data versioning and tracking
import numpy as np
import pandas as pd

# Profiling libraries
import sweetviz as sv
from scipy import stats

# Database and catalog
from sqlalchemy import (
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ydata_profiling import ProfileReport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


@dataclass
class DataProfile:
    """Comprehensive data profile"""

    dataset_id: str
    dataset_name: str
    timestamp: datetime
    shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    memory_usage: int
    statistics: dict[str, Any]
    missing_values: dict[str, float]
    unique_values: dict[str, int]
    correlations: dict[str, float]
    outliers: dict[str, int]
    distributions: dict[str, dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    checksum: str | None = None


@dataclass
class DataLineage:
    """Data lineage information"""

    lineage_id: str
    dataset_id: str
    source_datasets: list[str]
    transformations: list[dict[str, Any]]
    timestamp: datetime
    created_by: str
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataVersion:
    """Data version information"""

    version_id: str
    dataset_id: str
    version_number: str
    timestamp: datetime
    checksum: str
    changes: list[dict[str, Any]]
    stats_diff: dict[str, Any]
    schema_changes: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class DataProfiler:
    """Comprehensive data profiling engine"""

    def __init__(self, output_dir: str = "./data_profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiles = {}
        self.profile_history = []

    def profile_dataset(
        self, df: pd.DataFrame, dataset_name: str, detailed: bool = True
    ) -> DataProfile:
        """Generate comprehensive data profile"""
        dataset_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Basic information
        shape = df.shape
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        memory_usage = df.memory_usage(deep=True).sum()

        # Statistical summary
        statistics = self._compute_statistics(df)

        # Missing values analysis
        missing_values = (df.isnull().sum() / len(df)).to_dict()

        # Unique values
        unique_values = df.nunique().to_dict()

        # Correlations
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = {}
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    if col1 < col2:  # Upper triangle only
                        correlations[f"{col1}_vs_{col2}"] = corr_matrix.loc[col1, col2]

        # Outliers
        outliers = self._detect_outliers(df)

        # Distributions
        distributions = self._analyze_distributions(df) if detailed else {}

        # Calculate checksum
        checksum = self._calculate_checksum(df)

        profile = DataProfile(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            timestamp=timestamp,
            shape=shape,
            columns=columns,
            dtypes=dtypes,
            memory_usage=memory_usage,
            statistics=statistics,
            missing_values=missing_values,
            unique_values=unique_values,
            correlations=correlations,
            outliers=outliers,
            distributions=distributions,
            checksum=checksum,
            metadata={"profiling_method": "comprehensive", "detailed": detailed},
        )

        # Store profile
        self.profiles[dataset_id] = profile
        self.profile_history.append(profile)

        return profile

    def _compute_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute statistical summaries"""
        stats = {}

        for col in df.columns:
            col_stats = {}

            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75)),
                    "skewness": float(df[col].skew()),
                    "kurtosis": float(df[col].kurt()),
                }
            elif pd.api.types.is_categorical_dtype(
                df[col]
            ) or pd.api.types.is_object_dtype(df[col]):
                value_counts = df[col].value_counts()
                col_stats = {
                    "unique": int(df[col].nunique()),
                    "top": str(value_counts.index[0])
                    if len(value_counts) > 0
                    else None,
                    "freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "top_5": value_counts.head(5).to_dict(),
                }
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_stats = {
                    "min": str(df[col].min()),
                    "max": str(df[col].max()),
                    "range_days": (df[col].max() - df[col].min()).days,
                }

            stats[col] = col_stats

        return stats

    def _detect_outliers(self, df: pd.DataFrame) -> dict[str, int]:
        """Detect outliers using multiple methods"""
        outliers = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()

            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()

            outliers[col] = int(outlier_count)

        return outliers

    def _analyze_distributions(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Analyze data distributions"""
        distributions = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()

            if len(data) < 20:
                continue

            dist_info = {"type": "numeric", "normality_test": {}, "best_fit": None}

            # Normality tests
            try:
                statistic, p_value = stats.normaltest(data)
                dist_info["normality_test"] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05,
                }
            except:
                pass

            # Find best fitting distribution
            dist_names = ["norm", "expon", "uniform", "gamma", "beta"]
            best_dist = None
            best_ks_stat = np.inf

            for dist_name in dist_names:
                try:
                    dist = getattr(stats, dist_name)
                    params = dist.fit(data)
                    ks_stat, _ = stats.kstest(data, lambda x: dist.cdf(x, *params))

                    if ks_stat < best_ks_stat:
                        best_ks_stat = ks_stat
                        best_dist = dist_name
                except:
                    continue

            dist_info["best_fit"] = best_dist
            distributions[col] = dist_info

        return distributions

    def _calculate_checksum(self, df: pd.DataFrame) -> str:
        """Calculate dataset checksum"""
        # Create a string representation of the data
        data_str = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.sha256(data_str).hexdigest()

    def generate_html_report(self, profile: DataProfile) -> str:
        """Generate interactive HTML report using pandas-profiling"""
        # Load data if needed (for pandas-profiling)
        # This is a placeholder - in real use, you'd pass the actual DataFrame

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report - {profile.dataset_name}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .info-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }}
                .info-card h3 {{ margin-top: 0; color: #2c3e50; }}
                .stats-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .stats-table th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
                .stats-table td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                .stats-table tr:hover {{ background: #f5f5f5; }}
                .warning {{ background: #fff3cd; border-left-color: #ffc107; }}
                .error {{ background: #f8d7da; border-left-color: #dc3545; }}
                .success {{ background: #d4edda; border-left-color: #28a745; }}
                .chart-container {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Data Profile Report: {profile.dataset_name}</h1>
                <p><strong>Generated:</strong> {profile.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Dataset ID:</strong> {profile.dataset_id}</p>
                <p><strong>Checksum:</strong> <code>{profile.checksum[:16]}...</code></p>

                <h2>Overview</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>Shape</h3>
                        <p>Rows: {profile.shape[0]:,}</p>
                        <p>Columns: {profile.shape[1]}</p>
                    </div>
                    <div class="info-card">
                        <h3>Memory Usage</h3>
                        <p>{profile.memory_usage / 1024 / 1024:.2f} MB</p>
                    </div>
                    <div class="info-card">
                        <h3>Data Types</h3>
                        <p>Numeric: {sum(1 for dt in profile.dtypes.values() if "int" in dt or "float" in dt)}</p>
                        <p>Text: {sum(1 for dt in profile.dtypes.values() if "object" in dt or "string" in dt)}</p>
                        <p>DateTime: {sum(1 for dt in profile.dtypes.values() if "datetime" in dt)}</p>
                    </div>
                    <div class="info-card warning">
                        <h3>Missing Values</h3>
                        <p>Columns with missing: {sum(1 for v in profile.missing_values.values() if v > 0)}</p>
                        <p>Max missing: {max(profile.missing_values.values()) * 100:.1f}%</p>
                    </div>
                </div>

                <h2>Column Statistics</h2>
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Type</th>
                            <th>Missing %</th>
                            <th>Unique</th>
                            <th>Outliers</th>
                            <th>Key Stats</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for col in profile.columns:
            dtype = profile.dtypes.get(col, "unknown")
            missing_pct = profile.missing_values.get(col, 0) * 100
            unique = profile.unique_values.get(col, 0)
            outliers = profile.outliers.get(col, 0)

            # Get key stats for column
            col_stats = profile.statistics.get(col, {})
            if "mean" in col_stats:
                key_stats = f"μ={col_stats['mean']:.2f}, σ={col_stats['std']:.2f}"
            elif "top" in col_stats:
                key_stats = f"Top: {col_stats['top']} ({col_stats['freq']})"
            else:
                key_stats = "-"

            row_class = ""
            if missing_pct > 50:
                row_class = "error"
            elif missing_pct > 20:
                row_class = "warning"

            html += f"""
                        <tr class="{row_class}">
                            <td><strong>{col}</strong></td>
                            <td>{dtype}</td>
                            <td>{missing_pct:.1f}%</td>
                            <td>{unique:,}</td>
                            <td>{outliers}</td>
                            <td>{key_stats}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>

                <h2>Correlations</h2>
                <div class="chart-container">
                    <p>Top correlations:</p>
                    <ul>
        """

        # Add top correlations
        sorted_corr = sorted(
            profile.correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )[:10]
        for pair, corr in sorted_corr:
            if abs(corr) > 0.5:
                html += f"<li>{pair}: <strong>{corr:.3f}</strong></li>"

        html += """
                    </ul>
                </div>

                <h2>Data Quality Summary</h2>
                <div class="info-grid">
        """

        # Calculate quality metrics
        completeness = (1 - np.mean(list(profile.missing_values.values()))) * 100
        uniqueness = (
            np.mean([u / profile.shape[0] for u in profile.unique_values.values()])
            * 100
        )

        quality_class = (
            "success"
            if completeness > 90
            else "warning"
            if completeness > 70
            else "error"
        )
        html += f"""
                    <div class="info-card {quality_class}">
                        <h3>Completeness</h3>
                        <p>{completeness:.1f}%</p>
                    </div>
        """

        html += f"""
                    <div class="info-card">
                        <h3>Uniqueness</h3>
                        <p>{uniqueness:.1f}%</p>
                    </div>
                    <div class="info-card">
                        <h3>Total Outliers</h3>
                        <p>{sum(profile.outliers.values()):,}</p>
                    </div>
                </div>

            </div>
        </body>
        </html>
        """

        return html

    def generate_pandas_profile(
        self, df: pd.DataFrame, dataset_name: str
    ) -> ProfileReport:
        """Generate detailed report using pandas-profiling"""
        profile = ProfileReport(
            df,
            title=f"Data Profile Report - {dataset_name}",
            explorative=True,
            dark_mode=False,
            orange_mode=False,
        )
        return profile

    def generate_sweetviz_report(
        self, df: pd.DataFrame, dataset_name: str, target_column: str | None = None
    ):
        """Generate EDA report using Sweetviz"""
        if target_column and target_column in df.columns:
            report = sv.analyze(df, target_feat=target_column)
        else:
            report = sv.analyze(df)

        output_path = self.output_dir / f"{dataset_name}_sweetviz.html"
        report.show_html(str(output_path), open_browser=False)
        return output_path

    def compare_datasets(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name1: str = "Dataset 1",
        name2: str = "Dataset 2",
    ) -> dict[str, Any]:
        """Compare two datasets"""
        profile1 = self.profile_dataset(df1, name1, detailed=False)
        profile2 = self.profile_dataset(df2, name2, detailed=False)

        comparison = {
            "shape_diff": {
                "rows": profile2.shape[0] - profile1.shape[0],
                "cols": profile2.shape[1] - profile1.shape[1],
            },
            "columns_added": list(set(profile2.columns) - set(profile1.columns)),
            "columns_removed": list(set(profile1.columns) - set(profile2.columns)),
            "common_columns": list(set(profile1.columns) & set(profile2.columns)),
            "dtype_changes": {},
            "stats_comparison": {},
        }

        # Compare data types
        for col in comparison["common_columns"]:
            if profile1.dtypes.get(col) != profile2.dtypes.get(col):
                comparison["dtype_changes"][col] = {
                    "from": profile1.dtypes.get(col),
                    "to": profile2.dtypes.get(col),
                }

        # Compare statistics
        for col in comparison["common_columns"]:
            stats1 = profile1.statistics.get(col, {})
            stats2 = profile2.statistics.get(col, {})

            if "mean" in stats1 and "mean" in stats2:
                comparison["stats_comparison"][col] = {
                    "mean_diff": stats2["mean"] - stats1["mean"],
                    "std_diff": stats2["std"] - stats1["std"],
                    "missing_diff": profile2.missing_values.get(col, 0)
                    - profile1.missing_values.get(col, 0),
                }

        return comparison

    def save_profile(self, profile: DataProfile, format: str = "json"):
        """Save profile to file"""
        output_file = (
            self.output_dir / f"{profile.dataset_name}_{profile.dataset_id}.{format}"
        )

        if format == "json":
            with open(output_file, "w") as f:
                json.dump(asdict(profile), f, default=str, indent=2)
        elif format == "pickle":
            with open(output_file, "wb") as f:
                pickle.dump(profile, f)
        elif format == "html":
            html = self.generate_html_report(profile)
            with open(output_file, "w") as f:
                f.write(html)

        logger.info(f"Profile saved to {output_file}")
        return output_file


class DataLineageTracker:
    """Track data lineage and transformations"""

    def __init__(self, db_path: str = "./data_lineage.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.lineage_graph = {}

    def track_transformation(
        self,
        input_datasets: list[str],
        output_dataset: str,
        transformation: dict[str, Any],
        created_by: str = "system",
    ) -> DataLineage:
        """Track a data transformation"""
        lineage_id = str(uuid.uuid4())
        timestamp = datetime.now()

        lineage = DataLineage(
            lineage_id=lineage_id,
            dataset_id=output_dataset,
            source_datasets=input_datasets,
            transformations=[transformation],
            timestamp=timestamp,
            created_by=created_by,
            version="1.0",
        )

        # Update lineage graph
        self.lineage_graph[output_dataset] = {
            "sources": input_datasets,
            "transformation": transformation,
            "timestamp": timestamp,
        }

        # Store in database
        self._store_lineage(lineage)

        return lineage

    def _store_lineage(self, lineage: DataLineage):
        """Store lineage in database"""
        # Convert to JSON for storage
        lineage_record = {
            "lineage_id": lineage.lineage_id,
            "dataset_id": lineage.dataset_id,
            "source_datasets": json.dumps(lineage.source_datasets),
            "transformations": json.dumps(lineage.transformations),
            "timestamp": lineage.timestamp,
            "created_by": lineage.created_by,
            "version": lineage.version,
            "metadata": json.dumps(lineage.metadata),
        }

        # Store in SQLite (simplified version)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_lineage (
                lineage_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                source_datasets TEXT,
                transformations TEXT,
                timestamp DATETIME,
                created_by TEXT,
                version TEXT,
                metadata TEXT
            )
        """)

        cursor.execute(
            """
            INSERT INTO data_lineage VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            tuple(lineage_record.values()),
        )

        conn.commit()
        conn.close()

    def get_lineage(self, dataset_id: str) -> DataLineage | None:
        """Get lineage for a dataset"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM data_lineage WHERE dataset_id = ?
        """,
            (dataset_id,),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return DataLineage(
                lineage_id=result[0],
                dataset_id=result[1],
                source_datasets=json.loads(result[2]),
                transformations=json.loads(result[3]),
                timestamp=datetime.fromisoformat(result[4]),
                created_by=result[5],
                version=result[6],
                metadata=json.loads(result[7]),
            )

        return None

    def get_upstream_datasets(self, dataset_id: str, max_depth: int = 5) -> set[str]:
        """Get all upstream datasets"""
        upstream = set()
        to_process = [dataset_id]
        depth = 0

        while to_process and depth < max_depth:
            current = to_process.pop(0)
            lineage = self.get_lineage(current)

            if lineage:
                for source in lineage.source_datasets:
                    upstream.add(source)
                    to_process.append(source)

            depth += 1

        return upstream

    def visualize_lineage(self, dataset_id: str) -> str:
        """Generate lineage visualization"""
        import graphviz

        dot = graphviz.Digraph(comment=f"Data Lineage for {dataset_id}")
        dot.attr(rankdir="LR")

        # Add nodes and edges
        processed = set()
        to_process = [dataset_id]

        while to_process:
            current = to_process.pop(0)
            if current in processed:
                continue

            processed.add(current)
            dot.node(current, current)

            lineage = self.get_lineage(current)
            if lineage:
                for source in lineage.source_datasets:
                    dot.node(source, source)
                    dot.edge(source, current)
                    to_process.append(source)

        return dot.source


class DataVersionManager:
    """Manage data versions and track changes"""

    def __init__(self, storage_path: str = "./data_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.versions = {}
        self.current_versions = {}

    def create_version(
        self, df: pd.DataFrame, dataset_id: str, version_number: str | None = None
    ) -> DataVersion:
        """Create a new version of the dataset"""
        if version_number is None:
            # Auto-increment version
            current = self.current_versions.get(dataset_id, "0.0.0")
            parts = current.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            version_number = ".".join(parts)

        version_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Calculate checksum
        checksum = hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()

        # Detect changes from previous version
        changes = []
        stats_diff = {}
        schema_changes = []

        if dataset_id in self.current_versions:
            prev_version = self.versions.get(
                f"{dataset_id}_{self.current_versions[dataset_id]}"
            )
            if prev_version:
                changes, stats_diff, schema_changes = self._detect_changes(
                    df, prev_version
                )

        version = DataVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            version_number=version_number,
            timestamp=timestamp,
            checksum=checksum,
            changes=changes,
            stats_diff=stats_diff,
            schema_changes=schema_changes,
        )

        # Store version
        self.versions[f"{dataset_id}_{version_number}"] = version
        self.current_versions[dataset_id] = version_number

        # Save data snapshot
        self._save_snapshot(df, dataset_id, version_number)

        return version

    def _detect_changes(
        self, df: pd.DataFrame, prev_version: DataVersion
    ) -> tuple[list, dict, list]:
        """Detect changes from previous version"""
        changes = []
        stats_diff = {}
        schema_changes = []

        # Load previous data
        prev_df = self._load_snapshot(
            prev_version.dataset_id, prev_version.version_number
        )

        if prev_df is not None:
            # Detect schema changes
            new_cols = set(df.columns) - set(prev_df.columns)
            removed_cols = set(prev_df.columns) - set(df.columns)

            if new_cols:
                schema_changes.append(
                    {"type": "columns_added", "columns": list(new_cols)}
                )
            if removed_cols:
                schema_changes.append(
                    {"type": "columns_removed", "columns": list(removed_cols)}
                )

            # Detect row changes
            row_diff = len(df) - len(prev_df)
            if row_diff != 0:
                changes.append(
                    {
                        "type": "row_count_change",
                        "from": len(prev_df),
                        "to": len(df),
                        "difference": row_diff,
                    }
                )

            # Calculate statistics diff
            common_cols = set(df.columns) & set(prev_df.columns)
            for col in common_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats_diff[col] = {
                        "mean_change": df[col].mean() - prev_df[col].mean(),
                        "std_change": df[col].std() - prev_df[col].std(),
                    }

        return changes, stats_diff, schema_changes

    def _save_snapshot(self, df: pd.DataFrame, dataset_id: str, version_number: str):
        """Save data snapshot"""
        file_path = self.storage_path / f"{dataset_id}_{version_number}.parquet"
        df.to_parquet(file_path)

    def _load_snapshot(
        self, dataset_id: str, version_number: str
    ) -> pd.DataFrame | None:
        """Load data snapshot"""
        file_path = self.storage_path / f"{dataset_id}_{version_number}.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None

    def get_version(
        self, dataset_id: str, version_number: str
    ) -> DataVersion | None:
        """Get specific version"""
        return self.versions.get(f"{dataset_id}_{version_number}")

    def list_versions(self, dataset_id: str) -> list[DataVersion]:
        """List all versions for a dataset"""
        versions = []
        for key, version in self.versions.items():
            if version.dataset_id == dataset_id:
                versions.append(version)
        return sorted(versions, key=lambda x: x.timestamp)

    def rollback(self, dataset_id: str, version_number: str) -> pd.DataFrame | None:
        """Rollback to a specific version"""
        df = self._load_snapshot(dataset_id, version_number)
        if df is not None:
            self.current_versions[dataset_id] = version_number
            logger.info(f"Rolled back {dataset_id} to version {version_number}")
        return df


class DataCatalog:
    """Data catalog for dataset discovery and documentation"""

    def __init__(self, catalog_path: str = "./data_catalog.db"):
        self.catalog_path = catalog_path
        self.datasets = {}
        self._init_catalog()

    def _init_catalog(self):
        """Initialize catalog database"""
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                owner TEXT,
                created_date DATETIME,
                updated_date DATETIME,
                tags TEXT,
                schema_info TEXT,
                quality_score REAL,
                usage_count INTEGER,
                metadata TEXT
            )
        """)

        conn.commit()
        conn.close()

    def register_dataset(
        self,
        dataset_id: str,
        name: str,
        description: str,
        owner: str,
        tags: list[str] = None,
        metadata: dict = None,
    ):
        """Register a dataset in the catalog"""
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                dataset_id,
                name,
                description,
                owner,
                datetime.now(),
                datetime.now(),
                json.dumps(tags or []),
                "{}",  # Schema info placeholder
                0.0,  # Quality score placeholder
                0,  # Usage count
                json.dumps(metadata or {}),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Dataset {name} registered in catalog")

    def search_datasets(
        self, query: str = None, tags: list[str] = None, owner: str = None
    ) -> list[dict[str, Any]]:
        """Search datasets in catalog"""
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        where_clauses = []
        params = []

        if query:
            where_clauses.append("(name LIKE ? OR description LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])

        if owner:
            where_clauses.append("owner = ?")
            params.append(owner)

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        cursor.execute(
            f"""
            SELECT * FROM datasets WHERE {where_clause}
        """,
            params,
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "dataset_id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "owner": row[3],
                    "created_date": row[4],
                    "updated_date": row[5],
                    "tags": json.loads(row[6]),
                    "quality_score": row[8],
                    "usage_count": row[9],
                }
            )

        conn.close()

        # Filter by tags if specified
        if tags:
            results = [r for r in results if any(t in r["tags"] for t in tags)]

        return results

    def update_quality_score(self, dataset_id: str, quality_score: float):
        """Update dataset quality score"""
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE datasets SET quality_score = ?, updated_date = ?
            WHERE dataset_id = ?
        """,
            (quality_score, datetime.now(), dataset_id),
        )

        conn.commit()
        conn.close()

    def increment_usage(self, dataset_id: str):
        """Increment dataset usage counter"""
        conn = sqlite3.connect(self.catalog_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE datasets SET usage_count = usage_count + 1, updated_date = ?
            WHERE dataset_id = ?
        """,
            (datetime.now(), dataset_id),
        )

        conn.commit()
        conn.close()


# Example usage
if __name__ == "__main__":
    # Create sample data
    df = pd.DataFrame(
        {
            "id": range(1000),
            "age": np.random.normal(35, 10, 1000),
            "salary": np.random.normal(50000, 15000, 1000),
            "department": np.random.choice(["IT", "HR", "Sales"], 1000),
            "performance": np.random.uniform(0, 100, 1000),
        }
    )

    # Data Profiling
    profiler = DataProfiler()
    profile = profiler.profile_dataset(df, "employee_data")
    print(f"Dataset shape: {profile.shape}")
    print(f"Memory usage: {profile.memory_usage / 1024 / 1024:.2f} MB")

    # Generate reports
    html_report = profiler.generate_html_report(profile)
    profiler.save_profile(profile, format="html")

    # Data Lineage
    lineage_tracker = DataLineageTracker()
    lineage = lineage_tracker.track_transformation(
        input_datasets=["raw_employees", "raw_departments"],
        output_dataset="employee_data",
        transformation={"type": "join", "operations": ["filter", "aggregate"]},
    )

    # Data Versioning
    version_manager = DataVersionManager()
    version = version_manager.create_version(df, "employee_data", "1.0.0")
    print(f"Created version: {version.version_number}")

    # Data Catalog
    catalog = DataCatalog()
    catalog.register_dataset(
        dataset_id="employee_data_v1",
        name="Employee Data",
        description="Employee information with performance metrics",
        owner="HR Department",
        tags=["hr", "employees", "performance"],
    )

    # Search catalog
    results = catalog.search_datasets(query="employee")
    print(f"Found {len(results)} datasets")
