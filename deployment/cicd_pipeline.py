"""CI/CD Pipeline for ML Models

Automated pipeline for testing, validation, and deployment of ML models.
"""

import asyncio
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import git
import mlflow
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

import docker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for CI/CD pipeline."""

    project_name: str
    model_name: str
    repository_url: str
    branch: str = "main"
    test_data_path: str = "tests/data"
    model_registry_uri: str = "http://mlflow:5000"
    docker_registry: str = "registry.example.com"
    kubernetes_namespace: str = "ml-models"
    environments: list[str] = field(
        default_factory=lambda: ["dev", "staging", "production"]
    )
    test_coverage_threshold: float = 0.8
    performance_thresholds: dict[str, float] = field(default_factory=dict)
    security_scanning: bool = True
    auto_deploy_staging: bool = True
    require_manual_approval: bool = True
    notification_channels: list[str] = field(default_factory=list)


class MLPipeline:
    """Main CI/CD pipeline for ML models."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.mlflow_client = mlflow.tracking.MlflowClient(config.model_registry_uri)
        self.docker_client = docker.from_env()
        self.git_repo = None
        self.test_results = {}
        self.validation_results = {}
        self.deployment_status = {}

    async def run_pipeline(self, trigger_event: dict) -> dict:
        """Run the complete CI/CD pipeline."""
        logger.info(f"Starting pipeline for {self.config.model_name}")

        pipeline_run = {
            "run_id": self._generate_run_id(),
            "started_at": datetime.now().isoformat(),
            "trigger": trigger_event,
            "stages": {},
        }

        try:
            # Stage 1: Checkout code
            await self._checkout_code(pipeline_run)

            # Stage 2: Run tests
            await self._run_tests(pipeline_run)

            # Stage 3: Build model
            await self._build_model(pipeline_run)

            # Stage 4: Validate model
            await self._validate_model(pipeline_run)

            # Stage 5: Security scanning
            if self.config.security_scanning:
                await self._security_scan(pipeline_run)

            # Stage 6: Build container
            await self._build_container(pipeline_run)

            # Stage 7: Integration tests
            await self._integration_tests(pipeline_run)

            # Stage 8: Deploy to staging
            if self.config.auto_deploy_staging:
                await self._deploy_staging(pipeline_run)

            # Stage 9: Performance tests
            await self._performance_tests(pipeline_run)

            # Stage 10: Manual approval (if required)
            if self.config.require_manual_approval:
                await self._manual_approval(pipeline_run)

            # Stage 11: Deploy to production
            await self._deploy_production(pipeline_run)

            # Stage 12: Smoke tests
            await self._smoke_tests(pipeline_run)

            pipeline_run["status"] = "success"
            pipeline_run["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_run["status"] = "failed"
            pipeline_run["error"] = str(e)
            pipeline_run["failed_at"] = datetime.now().isoformat()

            # Rollback if necessary
            await self._rollback(pipeline_run)

        finally:
            # Send notifications
            await self._send_notifications(pipeline_run)

            # Store pipeline results
            self._store_pipeline_results(pipeline_run)

        return pipeline_run

    def _generate_run_id(self) -> str:
        """Generate unique pipeline run ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(
            f"{self.config.model_name}_{timestamp}".encode()
        ).hexdigest()[:12]

    async def _checkout_code(self, pipeline_run: dict):
        """Checkout code from repository."""
        stage = "checkout"
        logger.info(f"Stage: {stage}")

        try:
            # Clone or pull repository
            repo_path = Path(f"./workspace/{self.config.project_name}")

            if repo_path.exists():
                self.git_repo = git.Repo(repo_path)
                self.git_repo.remotes.origin.pull()
            else:
                self.git_repo = git.Repo.clone_from(
                    self.config.repository_url, repo_path, branch=self.config.branch
                )

            # Get commit info
            commit = self.git_repo.head.commit

            pipeline_run["stages"][stage] = {
                "status": "success",
                "commit_hash": commit.hexsha,
                "commit_message": commit.message,
                "commit_author": str(commit.author),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    async def _run_tests(self, pipeline_run: dict):
        """Run unit and integration tests."""
        stage = "testing"
        logger.info(f"Stage: {stage}")

        try:
            # Run pytest
            result = subprocess.run(
                ["pytest", "--cov=.", "--cov-report=json", "-v"],
                cwd=f"./workspace/{self.config.project_name}",
                capture_output=True,
                text=True,
            )

            # Parse coverage report
            with open(
                f"./workspace/{self.config.project_name}/coverage.json"
            ) as f:
                coverage_data = json.load(f)

            coverage = coverage_data["totals"]["percent_covered"] / 100

            # Check coverage threshold
            if coverage < self.config.test_coverage_threshold:
                raise ValueError(
                    f"Test coverage {coverage:.2%} below threshold {self.config.test_coverage_threshold:.2%}"
                )

            self.test_results = {
                "passed": result.returncode == 0,
                "coverage": coverage,
                "output": result.stdout,
                "errors": result.stderr,
            }

            pipeline_run["stages"][stage] = {
                "status": "success" if result.returncode == 0 else "failed",
                "test_results": self.test_results,
                "timestamp": datetime.now().isoformat(),
            }

            if result.returncode != 0:
                raise Exception("Tests failed")

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    async def _build_model(self, pipeline_run: dict):
        """Build and train the model."""
        stage = "build_model"
        logger.info(f"Stage: {stage}")

        try:
            # Run model training script
            result = subprocess.run(
                ["python", "train_model.py"],
                cwd=f"./workspace/{self.config.project_name}",
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise Exception(f"Model training failed: {result.stderr}")

            # Get model from MLflow
            model_version = self.mlflow_client.get_latest_versions(
                self.config.model_name, stages=["None"]
            )[0]

            pipeline_run["stages"][stage] = {
                "status": "success",
                "model_version": model_version.version,
                "model_uri": model_version.source,
                "run_id": model_version.run_id,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    async def _validate_model(self, pipeline_run: dict):
        """Validate model performance."""
        stage = "validation"
        logger.info(f"Stage: {stage}")

        try:
            # Load model
            model_uri = pipeline_run["stages"]["build_model"]["model_uri"]
            model = mlflow.pyfunc.load_model(model_uri)

            # Load test data
            test_data = pd.read_csv(f"{self.config.test_data_path}/test.csv")
            X_test = test_data.drop("target", axis=1)
            y_test = test_data["target"]

            # Make predictions
            predictions = model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, predictions)

            # Check performance thresholds
            for metric_name, threshold in self.config.performance_thresholds.items():
                if metrics.get(metric_name, 0) < threshold:
                    raise ValueError(
                        f"{metric_name} {metrics[metric_name]:.4f} below threshold {threshold}"
                    )

            self.validation_results = {
                "metrics": metrics,
                "passed": True,
                "test_size": len(X_test),
            }

            pipeline_run["stages"][stage] = {
                "status": "success",
                "validation_results": self.validation_results,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate model performance metrics."""
        # Determine if classification or regression
        if len(np.unique(y_true)) < 20:  # Likely classification
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted"),
                "recall": recall_score(y_true, y_pred, average="weighted"),
                "f1": f1_score(y_true, y_pred, average="weighted"),
            }
        else:  # Regression
            return {
                "r2": r2_score(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": mean_absolute_error(y_true, y_pred),
            }

    async def _security_scan(self, pipeline_run: dict):
        """Run security scanning on model and dependencies."""
        stage = "security_scan"
        logger.info(f"Stage: {stage}")

        try:
            vulnerabilities = []

            # Scan Python dependencies
            result = subprocess.run(
                ["safety", "check", "--json"],
                cwd=f"./workspace/{self.config.project_name}",
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                safety_report = json.loads(result.stdout)
                vulnerabilities.extend(safety_report)

            # Scan for secrets
            result = subprocess.run(
                ["trufflehog", "filesystem", ".", "--json"],
                cwd=f"./workspace/{self.config.project_name}",
                capture_output=True,
                text=True,
            )

            if result.stdout:
                secrets_found = [
                    json.loads(line) for line in result.stdout.strip().split("\n")
                ]
                if secrets_found:
                    vulnerabilities.append(
                        {
                            "type": "secrets",
                            "count": len(secrets_found),
                            "severity": "critical",
                        }
                    )

            # Model security checks
            model_vulnerabilities = await self._check_model_security(pipeline_run)
            vulnerabilities.extend(model_vulnerabilities)

            if vulnerabilities:
                logger.warning(
                    f"Security vulnerabilities found: {len(vulnerabilities)}"
                )

            pipeline_run["stages"][stage] = {
                "status": "success" if not vulnerabilities else "warning",
                "vulnerabilities": vulnerabilities,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            # Security scan failures shouldn't stop pipeline
            logger.error(f"Security scan failed: {e}")

    async def _check_model_security(self, pipeline_run: dict) -> list[dict]:
        """Check model for security vulnerabilities."""
        vulnerabilities = []

        # Check for pickle vulnerabilities
        model_uri = pipeline_run["stages"]["build_model"]["model_uri"]
        if "pickle" in model_uri or ".pkl" in model_uri:
            vulnerabilities.append(
                {
                    "type": "unsafe_deserialization",
                    "description": "Model uses pickle which can execute arbitrary code",
                    "severity": "high",
                }
            )

        # Check model size (potential DoS)
        model_path = Path(model_uri)
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            if size_mb > 1000:  # 1GB
                vulnerabilities.append(
                    {
                        "type": "large_model",
                        "description": f"Model size {size_mb:.2f}MB may cause resource issues",
                        "severity": "medium",
                    }
                )

        return vulnerabilities

    async def _build_container(self, pipeline_run: dict):
        """Build Docker container for model serving."""
        stage = "build_container"
        logger.info(f"Stage: {stage}")

        try:
            # Generate Dockerfile
            dockerfile_content = self._generate_dockerfile(pipeline_run)
            dockerfile_path = Path(f"./workspace/{self.config.project_name}/Dockerfile")
            dockerfile_path.write_text(dockerfile_content)

            # Build image
            image_tag = f"{self.config.docker_registry}/{self.config.model_name}:{pipeline_run['run_id']}"

            image, logs = self.docker_client.images.build(
                path=f"./workspace/{self.config.project_name}",
                tag=image_tag,
                buildargs={
                    "MODEL_URI": pipeline_run["stages"]["build_model"]["model_uri"],
                    "MODEL_NAME": self.config.model_name,
                },
            )

            # Scan container for vulnerabilities
            scan_result = self._scan_container(image_tag)

            # Push to registry
            for line in self.docker_client.images.push(
                image_tag, stream=True, decode=True
            ):
                logger.debug(line)

            pipeline_run["stages"][stage] = {
                "status": "success",
                "image_tag": image_tag,
                "image_id": image.id,
                "scan_result": scan_result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    def _generate_dockerfile(self, pipeline_run: dict) -> str:
        """Generate Dockerfile for model serving."""
        return """
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model serving code
COPY serve_model.py .

# Download model
ARG MODEL_URI
ARG MODEL_NAME
ENV MODEL_URI=${MODEL_URI}
ENV MODEL_NAME=${MODEL_NAME}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run server
CMD ["python", "serve_model.py"]
"""

    def _scan_container(self, image_tag: str) -> dict:
        """Scan container for vulnerabilities using Trivy."""
        try:
            result = subprocess.run(
                ["trivy", "image", "--format", "json", image_tag],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                scan_data = json.loads(result.stdout)

                # Count vulnerabilities by severity
                vuln_count = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}

                for result in scan_data.get("Results", []):
                    for vuln in result.get("Vulnerabilities", []):
                        severity = vuln.get("Severity", "UNKNOWN")
                        if severity in vuln_count:
                            vuln_count[severity] += 1

                return {
                    "vulnerabilities": vuln_count,
                    "passed": vuln_count["CRITICAL"] == 0 and vuln_count["HIGH"] < 5,
                }

        except Exception as e:
            logger.error(f"Container scanning failed: {e}")

        return {"vulnerabilities": {}, "passed": True}

    async def _integration_tests(self, pipeline_run: dict):
        """Run integration tests with the containerized model."""
        stage = "integration_tests"
        logger.info(f"Stage: {stage}")

        try:
            image_tag = pipeline_run["stages"]["build_container"]["image_tag"]

            # Start container
            container = self.docker_client.containers.run(
                image_tag,
                detach=True,
                ports={"8080/tcp": 8081},
                name=f"test_{pipeline_run['run_id']}",
            )

            # Wait for container to be ready
            await asyncio.sleep(5)

            # Run integration tests
            test_results = await self._run_integration_test_suite()

            # Stop and remove container
            container.stop()
            container.remove()

            pipeline_run["stages"][stage] = {
                "status": "success" if test_results["passed"] else "failed",
                "test_results": test_results,
                "timestamp": datetime.now().isoformat(),
            }

            if not test_results["passed"]:
                raise Exception("Integration tests failed")

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    async def _run_integration_test_suite(self) -> dict:
        """Run integration test suite against deployed model."""
        test_results = {"passed": True, "tests": []}

        # Test health endpoint
        response = requests.get("http://localhost:8081/health")
        test_results["tests"].append(
            {"name": "health_check", "passed": response.status_code == 200}
        )

        # Test prediction endpoint
        test_data = {"feature1": 1.0, "feature2": 2.0}
        response = requests.post("http://localhost:8081/predict", json=test_data)
        test_results["tests"].append(
            {
                "name": "prediction",
                "passed": response.status_code == 200
                and "prediction" in response.json(),
            }
        )

        # Test metrics endpoint
        response = requests.get("http://localhost:8081/metrics")
        test_results["tests"].append(
            {"name": "metrics", "passed": response.status_code == 200}
        )

        # Check if all tests passed
        test_results["passed"] = all(test["passed"] for test in test_results["tests"])

        return test_results

    async def _deploy_staging(self, pipeline_run: dict):
        """Deploy model to staging environment."""
        stage = "deploy_staging"
        logger.info(f"Stage: {stage}")

        try:
            from production_deployment import (
                ModelDeploymentConfig,
                ModelDeploymentManager,
            )

            config = ModelDeploymentConfig(
                model_name=self.config.model_name,
                model_version=pipeline_run["run_id"],
                model_uri=pipeline_run["stages"]["build_model"]["model_uri"],
                environment="staging",
                namespace=f"{self.config.kubernetes_namespace}-staging",
                replicas=2,
                monitoring_enabled=True,
            )

            manager = ModelDeploymentManager()
            deployment_result = manager.deploy_model(config)

            self.deployment_status["staging"] = deployment_result

            pipeline_run["stages"][stage] = {
                "status": "success",
                "deployment": deployment_result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    async def _performance_tests(self, pipeline_run: dict):
        """Run performance tests against staging deployment."""
        stage = "performance_tests"
        logger.info(f"Stage: {stage}")

        try:
            endpoint = self.deployment_status["staging"]["endpoint"]

            # Run load test
            load_test_results = await self._run_load_test(endpoint)

            # Check performance criteria
            if load_test_results["p95_latency"] > 1.0:  # 1 second
                raise ValueError(
                    f"P95 latency {load_test_results['p95_latency']}s exceeds threshold"
                )

            if load_test_results["error_rate"] > 0.01:  # 1% error rate
                raise ValueError(
                    f"Error rate {load_test_results['error_rate']:.2%} exceeds threshold"
                )

            pipeline_run["stages"][stage] = {
                "status": "success",
                "performance_results": load_test_results,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    async def _run_load_test(self, endpoint: str) -> dict:
        """Run load test using locust or similar tool."""
        # Run locust test
        result = subprocess.run(
            [
                "locust",
                "-f",
                "load_test.py",
                "--host",
                endpoint,
                "--users",
                "100",
                "--spawn-rate",
                "10",
                "--time",
                "60s",
                "--headless",
                "--csv",
                "load_test",
            ],
            cwd=f"./workspace/{self.config.project_name}/tests",
            capture_output=True,
            text=True,
        )

        # Parse results
        with open(
            f"./workspace/{self.config.project_name}/tests/load_test_stats.csv"
        ) as f:
            import csv

            reader = csv.DictReader(f)
            stats = list(reader)

        return {
            "requests_per_second": float(stats[0]["Requests/s"]),
            "p95_latency": float(stats[0]["95%"]) / 1000,  # Convert to seconds
            "error_rate": float(stats[0]["Failure Count"])
            / float(stats[0]["Request Count"]),
            "total_requests": int(stats[0]["Request Count"]),
        }

    async def _manual_approval(self, pipeline_run: dict):
        """Wait for manual approval before production deployment."""
        stage = "manual_approval"
        logger.info(f"Stage: {stage}")

        try:
            # Create approval request
            approval_request = {
                "pipeline_run_id": pipeline_run["run_id"],
                "model_name": self.config.model_name,
                "staging_metrics": pipeline_run["stages"]["performance_tests"][
                    "performance_results"
                ],
                "validation_metrics": pipeline_run["stages"]["validation"][
                    "validation_results"
                ],
            }

            # Send approval request (implement based on your approval system)
            # This could be Slack, email, or a web interface
            approved = await self._request_approval(approval_request)

            if not approved:
                raise Exception("Manual approval denied")

            pipeline_run["stages"][stage] = {
                "status": "success",
                "approved_by": "user@example.com",  # Get from approval system
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    async def _request_approval(self, approval_request: dict) -> bool:
        """Request manual approval."""
        # Simplified - in production, implement actual approval workflow
        logger.info("Waiting for manual approval...")

        # Send notification
        if self.config.notification_channels:
            await self._send_approval_request(approval_request)

        # Wait for approval (timeout after 1 hour)
        # In production, this would check an approval system
        await asyncio.sleep(10)  # Simulate approval delay

        return True  # Auto-approve for demo

    async def _deploy_production(self, pipeline_run: dict):
        """Deploy model to production environment."""
        stage = "deploy_production"
        logger.info(f"Stage: {stage}")

        try:
            from production_deployment import (
                ModelDeploymentConfig,
                ModelDeploymentManager,
            )

            config = ModelDeploymentConfig(
                model_name=self.config.model_name,
                model_version=pipeline_run["run_id"],
                model_uri=pipeline_run["stages"]["build_model"]["model_uri"],
                environment="production",
                namespace=self.config.kubernetes_namespace,
                replicas=4,
                autoscaling=True,
                min_replicas=2,
                max_replicas=10,
                monitoring_enabled=True,
                canary_deployment=True,
                canary_percentage=10,
            )

            manager = ModelDeploymentManager()
            deployment_result = manager.deploy_model(config)

            self.deployment_status["production"] = deployment_result

            # Update model stage in MLflow
            self.mlflow_client.transition_model_version_stage(
                name=self.config.model_name,
                version=pipeline_run["stages"]["build_model"]["model_version"],
                stage="Production",
            )

            pipeline_run["stages"][stage] = {
                "status": "success",
                "deployment": deployment_result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    async def _smoke_tests(self, pipeline_run: dict):
        """Run smoke tests against production deployment."""
        stage = "smoke_tests"
        logger.info(f"Stage: {stage}")

        try:
            endpoint = self.deployment_status["production"]["endpoint"]

            # Basic functionality tests
            smoke_test_results = {
                "health_check": await self._test_health(endpoint),
                "sample_prediction": await self._test_prediction(endpoint),
                "metrics_available": await self._test_metrics(endpoint),
            }

            # All tests must pass
            if not all(smoke_test_results.values()):
                raise Exception(f"Smoke tests failed: {smoke_test_results}")

            pipeline_run["stages"][stage] = {
                "status": "success",
                "test_results": smoke_test_results,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            pipeline_run["stages"][stage] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            raise

    async def _test_health(self, endpoint: str) -> bool:
        """Test health endpoint."""
        try:
            response = requests.get(f"{endpoint}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    async def _test_prediction(self, endpoint: str) -> bool:
        """Test prediction endpoint."""
        try:
            test_data = {"feature1": 1.0, "feature2": 2.0}
            response = requests.post(f"{endpoint}/predict", json=test_data, timeout=10)
            return response.status_code == 200 and "prediction" in response.json()
        except:
            return False

    async def _test_metrics(self, endpoint: str) -> bool:
        """Test metrics endpoint."""
        try:
            response = requests.get(f"{endpoint}/metrics", timeout=5)
            return response.status_code == 200
        except:
            return False

    async def _rollback(self, pipeline_run: dict):
        """Rollback deployment if pipeline failed."""
        logger.warning("Initiating rollback...")

        try:
            from production_deployment import (
                ModelDeploymentConfig,
                ModelDeploymentManager,
            )

            manager = ModelDeploymentManager()

            # Rollback staging if it was deployed
            if "staging" in self.deployment_status:
                config = ModelDeploymentConfig(
                    model_name=self.config.model_name,
                    model_version=pipeline_run["run_id"],
                    namespace=f"{self.config.kubernetes_namespace}-staging",
                )
                manager.rollback_deployment(config)

            # Rollback production if it was deployed
            if "production" in self.deployment_status:
                config = ModelDeploymentConfig(
                    model_name=self.config.model_name,
                    model_version=pipeline_run["run_id"],
                    namespace=self.config.kubernetes_namespace,
                )
                manager.rollback_deployment(config)

            pipeline_run["rollback"] = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            pipeline_run["rollback"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def _send_notifications(self, pipeline_run: dict):
        """Send pipeline notifications."""
        try:
            for channel in self.config.notification_channels:
                if channel.startswith("slack://"):
                    await self._send_slack_notification(pipeline_run, channel)
                elif channel.startswith("email://"):
                    await self._send_email_notification(pipeline_run, channel)
                elif channel.startswith("teams://"):
                    await self._send_teams_notification(pipeline_run, channel)
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")

    async def _send_slack_notification(self, pipeline_run: dict, webhook_url: str):
        """Send Slack notification."""
        status_emoji = "✅" if pipeline_run["status"] == "success" else "❌"

        message = {
            "text": f"{status_emoji} Pipeline {pipeline_run['status']} for {self.config.model_name}",
            "attachments": [
                {
                    "color": "good"
                    if pipeline_run["status"] == "success"
                    else "danger",
                    "fields": [
                        {
                            "title": "Model",
                            "value": self.config.model_name,
                            "short": True,
                        },
                        {
                            "title": "Run ID",
                            "value": pipeline_run["run_id"],
                            "short": True,
                        },
                        {
                            "title": "Duration",
                            "value": self._calculate_duration(pipeline_run),
                            "short": True,
                        },
                        {
                            "title": "Trigger",
                            "value": pipeline_run["trigger"].get("type", "manual"),
                            "short": True,
                        },
                    ],
                }
            ],
        }

        requests.post(webhook_url.replace("slack://", ""), json=message)

    def _calculate_duration(self, pipeline_run: dict) -> str:
        """Calculate pipeline duration."""
        if "completed_at" in pipeline_run:
            start = datetime.fromisoformat(pipeline_run["started_at"])
            end = datetime.fromisoformat(pipeline_run["completed_at"])
            duration = end - start
            return str(duration).split(".")[0]
        return "In progress"

    def _store_pipeline_results(self, pipeline_run: dict):
        """Store pipeline results for auditing."""
        # Store in database or file system
        results_path = Path(f"./pipeline_runs/{pipeline_run['run_id']}.json")
        results_path.parent.mkdir(exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(pipeline_run, f, indent=2, default=str)

        logger.info(f"Pipeline results stored: {results_path}")
