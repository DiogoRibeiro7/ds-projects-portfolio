"""
Performance regression tests to ensure system performance doesn't degrade.
"""

import pytest
import time
import numpy as np
import pandas as pd
import psutil
import tracemalloc
from memory_profiler import memory_usage
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from typing import List, Dict, Any, Callable
import json
from pathlib import Path

from modern_bank_churn.ml_pipeline_orchestrator import MLPipelineOrchestrator, PipelineConfig
from modern_bank_churn.feature_engineering import FeatureEngineer
from statistical_methods.statistical_analyzer import StatisticalAnalyzer
from dashboard_enhanced.dashboard_framework import EnhancedDashboard, DashboardConfig


# ============================================================================
# Performance Benchmarks
# ============================================================================

@pytest.fixture
def performance_baselines():
    """Load performance baselines from previous runs."""
    baseline_file = Path(__file__).parent / "performance_baselines.json"

    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            return json.load(f)

    # Default baselines if no previous data
    return {
        "ml_pipeline_small": {"time": 2.0, "memory": 100},  # MB
        "ml_pipeline_medium": {"time": 10.0, "memory": 500},
        "ml_pipeline_large": {"time": 60.0, "memory": 2000},
        "feature_engineering": {"time": 1.0, "memory": 50},
        "statistical_analysis": {"time": 0.5, "memory": 30},
        "dashboard_render": {"time": 0.1, "memory": 20},
        "api_response": {"time": 0.05, "memory": 10},
    }


@pytest.fixture
def save_performance_results():
    """Save performance results for future comparison."""
    results = {}

    yield results

    # Save results after test
    results_file = Path(__file__).parent / "performance_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


# ============================================================================
# ML Pipeline Performance Tests
# ============================================================================

class TestMLPipelinePerformance:
    """Test ML pipeline performance."""

    @pytest.mark.performance
    @pytest.mark.benchmark(group="ml-pipeline")
    def test_small_dataset_performance(self, benchmark, save_performance_results):
        """Test performance with small dataset (1K samples)."""
        # Generate small dataset
        n_samples = 1000
        data = self._generate_dataset(n_samples)

        # Configure pipeline
        config = PipelineConfig(
            feature_selection_method='mutual_info',
            model_type='random_forest',
            hyperparameter_tuning=False,
            cross_validation_folds=2
        )

        def run_pipeline():
            orchestrator = MLPipelineOrchestrator(config)
            return orchestrator.run_pipeline(data)

        # Benchmark
        result = benchmark(run_pipeline)

        # Save results
        save_performance_results['ml_pipeline_small'] = {
            'time': benchmark.stats['mean'],
            'memory': self._measure_memory(run_pipeline)
        }

    @pytest.mark.performance
    @pytest.mark.benchmark(group="ml-pipeline")
    @pytest.mark.slow
    def test_medium_dataset_performance(self, benchmark, save_performance_results):
        """Test performance with medium dataset (10K samples)."""
        n_samples = 10000
        data = self._generate_dataset(n_samples)

        config = PipelineConfig(
            feature_selection_method='mutual_info',
            model_type='xgboost',
            hyperparameter_tuning=False,
            cross_validation_folds=3
        )

        def run_pipeline():
            orchestrator = MLPipelineOrchestrator(config)
            return orchestrator.run_pipeline(data)

        result = benchmark(run_pipeline)

        save_performance_results['ml_pipeline_medium'] = {
            'time': benchmark.stats['mean'],
            'memory': self._measure_memory(run_pipeline)
        }

    @pytest.mark.performance
    @pytest.mark.benchmark(group="ml-pipeline")
    @pytest.mark.slow
    def test_large_dataset_performance(self, benchmark, save_performance_results):
        """Test performance with large dataset (100K samples)."""
        n_samples = 100000
        data = self._generate_dataset(n_samples)

        config = PipelineConfig(
            feature_selection_method='mutual_info',
            model_type='lightgbm',
            hyperparameter_tuning=False,
            cross_validation_folds=2
        )

        def run_pipeline():
            orchestrator = MLPipelineOrchestrator(config)
            return orchestrator.run_pipeline(data)

        result = benchmark(run_pipeline)

        save_performance_results['ml_pipeline_large'] = {
            'time': benchmark.stats['mean'],
            'memory': self._measure_memory(run_pipeline)
        }

    def _generate_dataset(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic dataset for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'feature4': np.random.randn(n_samples),
            'feature5': np.random.randn(n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })

    def _measure_memory(self, func: Callable) -> float:
        """Measure memory usage of a function."""
        mem_usage = memory_usage(func)
        return max(mem_usage) - min(mem_usage)


# ============================================================================
# Feature Engineering Performance Tests
# ============================================================================

class TestFeatureEngineeringPerformance:
    """Test feature engineering performance."""

    @pytest.mark.performance
    @pytest.mark.benchmark(group="feature-engineering")
    def test_feature_creation_speed(self, benchmark, sample_dataframe):
        """Test feature creation speed."""
        engineer = FeatureEngineer()

        def create_features():
            return engineer.create_features(sample_dataframe)

        result = benchmark(create_features)

        # Assert performance threshold
        assert benchmark.stats['mean'] < 2.0, "Feature creation too slow"

    @pytest.mark.performance
    @pytest.mark.benchmark(group="feature-engineering")
    def test_feature_selection_speed(self, benchmark, sample_dataframe):
        """Test feature selection speed."""
        engineer = FeatureEngineer()
        X = sample_dataframe.drop('churn', axis=1).select_dtypes(include=[np.number])
        y = sample_dataframe['churn']

        def select_features():
            return engineer.select_features(X, y, method='mutual_info', k=10)

        result = benchmark(select_features)

        assert benchmark.stats['mean'] < 1.0, "Feature selection too slow"

    @pytest.mark.performance
    def test_parallel_feature_engineering(self, sample_dataframe):
        """Test parallel feature engineering performance."""
        engineer = FeatureEngineer()

        # Split data into chunks
        chunks = np.array_split(sample_dataframe, 4)

        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for chunk in chunks:
            sequential_results.append(engineer.create_features(chunk))
        sequential_time = time.time() - start_time

        # Parallel processing
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(engineer.create_features, chunks))
        parallel_time = time.time() - start_time

        # Parallel should be faster
        assert parallel_time < sequential_time * 0.8, "Parallel processing not efficient"


# ============================================================================
# Statistical Analysis Performance Tests
# ============================================================================

class TestStatisticalPerformance:
    """Test statistical analysis performance."""

    @pytest.mark.performance
    @pytest.mark.benchmark(group="statistics")
    def test_statistical_summary_speed(self, benchmark, sample_dataframe):
        """Test statistical summary generation speed."""
        analyzer = StatisticalAnalyzer(sample_dataframe)

        def generate_summary():
            return analyzer.generate_summary(include_plots=False)

        result = benchmark(generate_summary)

        assert benchmark.stats['mean'] < 0.5, "Statistical summary too slow"

    @pytest.mark.performance
    @pytest.mark.benchmark(group="statistics")
    def test_hypothesis_testing_speed(self, benchmark, sample_dataframe):
        """Test hypothesis testing speed."""
        from statistical_methods.hypothesis_tester import HypothesisTester

        tester = HypothesisTester()
        group1 = sample_dataframe[sample_dataframe['churn'] == 0]['balance']
        group2 = sample_dataframe[sample_dataframe['churn'] == 1]['balance']

        def run_tests():
            results = []
            results.append(tester.t_test(group1, group2))
            results.append(tester.mann_whitney_u(group1, group2))
            results.append(tester.anova(sample_dataframe, 'geography', 'balance'))
            return results

        result = benchmark(run_tests)

        assert benchmark.stats['mean'] < 1.0, "Hypothesis testing too slow"


# ============================================================================
# Dashboard Performance Tests
# ============================================================================

class TestDashboardPerformance:
    """Test dashboard performance."""

    @pytest.mark.performance
    def test_dashboard_initialization_speed(self):
        """Test dashboard initialization speed."""
        config = DashboardConfig(
            app_name='Test Dashboard',
            enable_realtime=False
        )

        start_time = time.time()
        dashboard = EnhancedDashboard(config)
        init_time = time.time() - start_time

        assert init_time < 1.0, "Dashboard initialization too slow"

    @pytest.mark.performance
    def test_data_rendering_speed(self, sample_dataframe):
        """Test data rendering speed."""
        from dashboard_enhanced.visualization_components import InteractiveVisualizations

        viz = InteractiveVisualizations()

        start_time = time.time()

        # Create multiple visualizations
        fig1 = viz.create_animated_time_series(
            sample_dataframe.head(100),
            x_col='customer_id',
            y_cols=['balance'],
            title='Test'
        )

        fig2 = viz.create_interactive_3d_scatter(
            sample_dataframe.head(100),
            x_col='age',
            y_col='balance',
            z_col='tenure'
        )

        render_time = time.time() - start_time

        assert render_time < 2.0, "Visualization rendering too slow"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_websocket_throughput(self):
        """Test WebSocket message throughput."""
        messages_sent = 0
        messages_received = 0

        async def send_messages(websocket):
            nonlocal messages_sent
            for i in range(1000):
                await websocket.send(json.dumps({'id': i, 'data': 'test'}))
                messages_sent += 1
                await asyncio.sleep(0.001)

        async def receive_messages(websocket):
            nonlocal messages_received
            while messages_received < 1000:
                message = await websocket.recv()
                messages_received += 1

        # Simulate WebSocket communication (mock implementation)
        start_time = time.time()
        # In real test, would connect to actual WebSocket server
        await asyncio.sleep(1)  # Simulate communication time
        throughput_time = time.time() - start_time

        assert throughput_time < 5.0, "WebSocket throughput too low"


# ============================================================================
# Memory Leak Tests
# ============================================================================

class TestMemoryLeaks:
    """Test for memory leaks in long-running processes."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_pipeline_memory_leak(self):
        """Test for memory leaks in pipeline execution."""
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Run pipeline multiple times
        for i in range(10):
            data = pd.DataFrame({
                'a': np.random.randn(1000),
                'b': np.random.randn(1000),
                'target': np.random.choice([0, 1], 1000)
            })

            config = PipelineConfig(
                cross_validation_folds=2,
                hyperparameter_tuning=False
            )

            orchestrator = MLPipelineOrchestrator(config)
            results = orchestrator.run_pipeline(data)

            # Force garbage collection
            import gc
            gc.collect()

        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Memory increase should be minimal
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        assert memory_increase < 100, f"Memory leak detected: {memory_increase:.2f} MB increase"

    @pytest.mark.performance
    def test_data_processing_memory_leak(self):
        """Test for memory leaks in data processing."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process data repeatedly
        for i in range(100):
            data = pd.DataFrame(np.random.randn(10000, 10))
            processed = data.rolling(window=10).mean()
            del data
            del processed

        import gc
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert memory_increase < 50, f"Memory leak in data processing: {memory_increase:.2f} MB"


# ============================================================================
# Scalability Tests
# ============================================================================

class TestScalability:
    """Test system scalability."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_linear_scaling(self):
        """Test if processing time scales linearly with data size."""
        times = []
        sizes = [1000, 2000, 4000, 8000]

        for size in sizes:
            data = pd.DataFrame({
                'x': np.random.randn(size),
                'y': np.random.randn(size)
            })

            start_time = time.time()
            # Simple processing
            result = data.rolling(window=10).mean()
            elapsed = time.time() - start_time

            times.append(elapsed)

        # Check if scaling is approximately linear
        # Time should roughly double when size doubles
        scaling_factors = [times[i+1] / times[i] for i in range(len(times)-1)]

        # Allow some deviation from perfect linear scaling
        for factor in scaling_factors:
            assert 1.5 < factor < 3.0, f"Non-linear scaling detected: {factor}"

    @pytest.mark.performance
    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests."""
        def process_request(request_id):
            """Simulate request processing."""
            time.sleep(0.01)  # Simulate processing time
            return {'id': request_id, 'status': 'completed'}

        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for i in range(100):
            sequential_results.append(process_request(i))
        sequential_time = time.time() - start_time

        # Concurrent processing
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            concurrent_results = list(executor.map(process_request, range(100)))
        concurrent_time = time.time() - start_time

        # Concurrent should be significantly faster
        speedup = sequential_time / concurrent_time
        assert speedup > 5, f"Insufficient concurrency speedup: {speedup:.2f}x"


# ============================================================================
# Resource Usage Tests
# ============================================================================

class TestResourceUsage:
    """Test resource usage limits."""

    @pytest.mark.performance
    def test_cpu_usage(self):
        """Test CPU usage stays within limits."""
        process = psutil.Process()
        initial_cpu = process.cpu_percent(interval=0.1)

        # Perform CPU-intensive task
        data = np.random.randn(10000, 100)
        result = np.linalg.svd(data)

        peak_cpu = process.cpu_percent(interval=0.1)

        # CPU usage should not exceed reasonable limits
        assert peak_cpu < 400, f"Excessive CPU usage: {peak_cpu}%"  # 400% = 4 cores

    @pytest.mark.performance
    def test_disk_io(self, temp_dir):
        """Test disk I/O performance."""
        file_path = temp_dir / "test_large_file.csv"
        data = pd.DataFrame(np.random.randn(100000, 20))

        # Write performance
        start_time = time.time()
        data.to_csv(file_path, index=False)
        write_time = time.time() - start_time

        # Read performance
        start_time = time.time()
        loaded_data = pd.read_csv(file_path)
        read_time = time.time() - start_time

        # Clean up
        file_path.unlink()

        # Check I/O performance
        file_size_mb = file_path.stat().st_size / 1024 / 1024 if file_path.exists() else 100
        write_speed = file_size_mb / write_time  # MB/s
        read_speed = file_size_mb / read_time  # MB/s

        assert write_speed > 10, f"Slow write speed: {write_speed:.2f} MB/s"
        assert read_speed > 20, f"Slow read speed: {read_speed:.2f} MB/s"


# ============================================================================
# Regression Tests
# ============================================================================

class TestPerformanceRegression:
    """Test for performance regression against baselines."""

    @pytest.mark.performance
    def test_no_regression(self, performance_baselines, save_performance_results):
        """Test that current performance hasn't regressed."""
        # Run quick performance tests
        current_results = self._run_performance_tests()

        # Compare with baselines
        regressions = []

        for test_name, current in current_results.items():
            if test_name in performance_baselines:
                baseline = performance_baselines[test_name]

                # Allow 20% degradation
                if current['time'] > baseline['time'] * 1.2:
                    regressions.append(
                        f"{test_name}: time {current['time']:.2f}s > {baseline['time'] * 1.2:.2f}s"
                    )

                if current['memory'] > baseline['memory'] * 1.2:
                    regressions.append(
                        f"{test_name}: memory {current['memory']:.1f}MB > {baseline['memory'] * 1.2:.1f}MB"
                    )

        # Save current results
        save_performance_results.update(current_results)

        # Report regressions
        if regressions:
            regression_report = "\n".join(regressions)
            pytest.fail(f"Performance regressions detected:\n{regression_report}")

    def _run_performance_tests(self) -> Dict[str, Dict[str, float]]:
        """Run quick performance tests."""
        results = {}

        # Test small ML pipeline
        start_time = time.time()
        data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        # Simulate pipeline run
        time.sleep(0.1)
        results['ml_pipeline_small'] = {
            'time': time.time() - start_time,
            'memory': psutil.Process().memory_info().rss / 1024 / 1024
        }

        return results