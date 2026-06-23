"""Production Deployment System

Comprehensive model deployment infrastructure with Kubernetes, monitoring, and auto-scaling.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import boto3
import mlflow
import numpy as np
import pandas as pd
import redis
from prometheus_client import Counter, Gauge, Histogram

import docker
from kubernetes import client, config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
prediction_counter = Counter(
    "model_predictions_total", "Total predictions", ["model", "version", "status"]
)
prediction_latency = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency",
    ["model", "version"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)
model_accuracy = Gauge("model_accuracy", "Model accuracy", ["model", "version"])
data_drift_score = Gauge("data_drift_score", "Data drift score", ["model", "feature"])
model_health = Gauge("model_health_status", "Model health status", ["model", "version"])
deployment_status = Gauge(
    "deployment_status", "Deployment status", ["model", "version", "environment"]
)
cache_hits = Counter("cache_hits_total", "Cache hit count", ["model"])
cache_misses = Counter("cache_misses_total", "Cache miss count", ["model"])
error_counter = Counter(
    "model_errors_total", "Model errors", ["model", "version", "error_type"]
)


@dataclass
class ModelDeploymentConfig:
    """Configuration for model deployment."""

    model_name: str
    model_version: str
    model_uri: str
    model_type: str = "sklearn"  # sklearn, tensorflow, pytorch, xgboost
    replicas: int = 2
    cpu_request: str = "100m"
    memory_request: str = "512Mi"
    cpu_limit: str = "1"
    memory_limit: str = "2Gi"
    gpu_enabled: bool = False
    gpu_count: int = 0
    autoscaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    environment: str = "production"  # staging, production
    namespace: str = "ml-models"
    monitoring_enabled: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 300  # seconds
    health_check_interval: int = 30  # seconds
    rollback_on_failure: bool = True
    canary_deployment: bool = False
    canary_percentage: int = 10
    a_b_testing: bool = False
    a_b_percentage: int = 50
    custom_metrics: dict[str, Any] = field(default_factory=dict)
    environment_variables: dict[str, str] = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)


class ModelDeploymentManager:
    """Manage model deployments to Kubernetes."""

    def __init__(self, config_file: str | None = None):
        """Initialize deployment manager."""
        if config_file:
            config.load_kube_config(config_file=config_file)
        else:
            try:
                config.load_incluster_config()  # For in-cluster
            except:
                config.load_kube_config()  # For local development

        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.batch_v1 = client.BatchV1Api()

        # Initialize monitoring
        self.monitor = ModelMonitor()

        # Initialize cache
        self.cache = redis.Redis(host="redis", port=6379, db=0)

        # MLflow client
        self.mlflow_client = mlflow.tracking.MlflowClient()

    def deploy_model(self, config: ModelDeploymentConfig) -> dict[str, Any]:
        """Deploy model to Kubernetes."""
        logger.info(
            f"Deploying model {config.model_name} version {config.model_version}"
        )

        try:
            # Create namespace if not exists
            self._ensure_namespace(config.namespace)

            # Create ConfigMap for model metadata
            configmap = self._create_configmap(config)
            logger.info(f"Created ConfigMap: {configmap.metadata.name}")

            # Create Secrets if needed
            if config.secrets:
                secrets = self._create_secrets(config)
                logger.info(f"Created Secrets: {secrets}")

            # Create PersistentVolumeClaim for model storage
            pvc = self._create_pvc(config)
            logger.info(f"Created PVC: {pvc.metadata.name}")

            # Build and push Docker image
            image_uri = self._build_and_push_image(config)
            logger.info(f"Built and pushed image: {image_uri}")

            # Create Deployment
            if config.canary_deployment:
                deployment = self._create_canary_deployment(config, image_uri)
            else:
                deployment = self._create_deployment(config, image_uri)
            logger.info(f"Created Deployment: {deployment.metadata.name}")

            # Create Service
            service = self._create_service(config)
            logger.info(f"Created Service: {service.metadata.name}")

            # Create Ingress
            ingress = self._create_ingress(config)
            logger.info(f"Created Ingress: {ingress.metadata.name}")

            # Create HPA if autoscaling enabled
            if config.autoscaling:
                hpa = self._create_hpa(config)
                logger.info(f"Created HPA: {hpa.metadata.name}")

            # Setup monitoring
            if config.monitoring_enabled:
                monitoring = self._setup_monitoring(config)
                logger.info(f"Setup monitoring: {monitoring}")

            # Wait for deployment to be ready
            ready = self._wait_for_deployment(config)

            if not ready and config.rollback_on_failure:
                logger.warning("Deployment failed, rolling back...")
                self.rollback_deployment(config)
                raise Exception("Deployment failed and rolled back")

            # Update deployment status metric
            deployment_status.labels(
                model=config.model_name,
                version=config.model_version,
                environment=config.environment,
            ).set(1)

            return {
                "status": "success",
                "configmap": configmap.metadata.name,
                "deployment": deployment.metadata.name,
                "service": service.metadata.name,
                "ingress": ingress.metadata.name,
                "endpoint": f"http://{ingress.spec.rules[0].host}/predict",
                "monitoring_dashboard": f"http://grafana.monitoring/d/{config.model_name}",
            }

        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            deployment_status.labels(
                model=config.model_name,
                version=config.model_version,
                environment=config.environment,
            ).set(0)
            raise

    def _ensure_namespace(self, namespace: str):
        """Ensure namespace exists."""
        try:
            self.v1.read_namespace(namespace)
        except client.exceptions.ApiException as e:
            if e.status == 404:
                body = client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
                self.v1.create_namespace(body)

    def _create_configmap(self, config: ModelDeploymentConfig):
        """Create ConfigMap for model configuration."""
        configmap_data = {
            "model_name": config.model_name,
            "model_version": config.model_version,
            "model_uri": config.model_uri,
            "model_type": config.model_type,
            "environment": config.environment,
            "cache_enabled": str(config.cache_enabled),
            "cache_ttl": str(config.cache_ttl),
            "monitoring_enabled": str(config.monitoring_enabled),
            "custom_metrics": json.dumps(config.custom_metrics),
        }

        configmap = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(
                name=f"{config.model_name}-config", namespace=config.namespace
            ),
            data=configmap_data,
        )

        try:
            return self.v1.create_namespaced_config_map(
                namespace=config.namespace, body=configmap
            )
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                return self.v1.patch_namespaced_config_map(
                    name=f"{config.model_name}-config",
                    namespace=config.namespace,
                    body=configmap,
                )
            raise

    def _create_secrets(self, config: ModelDeploymentConfig) -> list[str]:
        """Create Kubernetes secrets."""
        created_secrets = []

        for secret_name in config.secrets:
            # In production, fetch from secure vault
            secret_data = self._fetch_secret_from_vault(secret_name)

            secret = client.V1Secret(
                api_version="v1",
                kind="Secret",
                metadata=client.V1ObjectMeta(
                    name=secret_name, namespace=config.namespace
                ),
                data={k: v.encode("utf-8") for k, v in secret_data.items()},
            )

            try:
                self.v1.create_namespaced_secret(
                    namespace=config.namespace, body=secret
                )
                created_secrets.append(secret_name)
            except client.exceptions.ApiException as e:
                if e.status != 409:  # Ignore if already exists
                    raise

        return created_secrets

    def _create_pvc(self, config: ModelDeploymentConfig):
        """Create PersistentVolumeClaim for model storage."""
        pvc = client.V1PersistentVolumeClaim(
            api_version="v1",
            kind="PersistentVolumeClaim",
            metadata=client.V1ObjectMeta(
                name=f"{config.model_name}-pvc", namespace=config.namespace
            ),
            spec=client.V1PersistentVolumeClaimSpec(
                access_modes=["ReadWriteOnce"],
                resources=client.V1ResourceRequirements(requests={"storage": "10Gi"}),
                storage_class_name="standard",
            ),
        )

        try:
            return self.v1.create_namespaced_persistent_volume_claim(
                namespace=config.namespace, body=pvc
            )
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                return self.v1.read_namespaced_persistent_volume_claim(
                    name=f"{config.model_name}-pvc", namespace=config.namespace
                )
            raise

    def _build_and_push_image(self, config: ModelDeploymentConfig) -> str:
        """Build and push Docker image for model serving."""
        # Docker client
        docker_client = docker.from_env()

        # Generate Dockerfile
        dockerfile_content = self._generate_dockerfile(config)

        # Build image
        image_name = f"{config.model_name}:{config.model_version}"
        registry = "your-registry.azurecr.io"  # Replace with your registry
        full_image_name = f"{registry}/{image_name}"

        # Build context
        build_context = Path("./model_serving")
        build_context.mkdir(exist_ok=True)

        # Write Dockerfile
        with open(build_context / "Dockerfile", "w") as f:
            f.write(dockerfile_content)

        # Copy model serving code
        self._prepare_model_serving_code(config, build_context)

        # Build image
        image, logs = docker_client.images.build(
            path=str(build_context),
            tag=full_image_name,
            buildargs={
                "MODEL_NAME": config.model_name,
                "MODEL_VERSION": config.model_version,
                "MODEL_URI": config.model_uri,
            },
        )

        # Push to registry
        for line in docker_client.images.push(
            full_image_name, stream=True, decode=True
        ):
            logger.debug(line)

        return full_image_name

    def _generate_dockerfile(self, config: ModelDeploymentConfig) -> str:
        """Generate Dockerfile for model serving."""
        base_image = {
            "sklearn": "python:3.9-slim",
            "tensorflow": "tensorflow/serving:latest",
            "pytorch": "pytorch/pytorch:latest",
            "xgboost": "python:3.9-slim",
        }.get(config.model_type, "python:3.9-slim")

        dockerfile = f"""
FROM {base_image}

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model serving code
COPY . .

# Download model from MLflow
ARG MODEL_NAME
ARG MODEL_VERSION
ARG MODEL_URI

RUN python download_model.py --uri $MODEL_URI --output /app/model

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run server
CMD ["python", "model_server.py"]
"""
        return dockerfile

    def _create_deployment(self, config: ModelDeploymentConfig, image_uri: str):
        """Create Kubernetes deployment."""
        # Container configuration
        container = client.V1Container(
            name=f"{config.model_name}-container",
            image=image_uri,
            image_pull_policy="Always",
            ports=[
                client.V1ContainerPort(container_port=8080, name="http"),
                client.V1ContainerPort(container_port=9090, name="metrics"),
            ],
            env=self._create_env_vars(config),
            env_from=[
                client.V1EnvFromSource(
                    config_map_ref=client.V1ConfigMapEnvSource(
                        name=f"{config.model_name}-config"
                    )
                )
            ],
            resources=client.V1ResourceRequirements(
                requests={"cpu": config.cpu_request, "memory": config.memory_request},
                limits={"cpu": config.cpu_limit, "memory": config.memory_limit},
            ),
            volume_mounts=[
                client.V1VolumeMount(name="model-storage", mount_path="/app/model")
            ],
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(path="/health", port=8080),
                initial_delay_seconds=30,
                period_seconds=config.health_check_interval,
                timeout_seconds=5,
                failure_threshold=3,
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(path="/ready", port=8080),
                initial_delay_seconds=5,
                period_seconds=10,
                timeout_seconds=5,
                failure_threshold=3,
            ),
        )

        # Add GPU resources if enabled
        if config.gpu_enabled:
            container.resources.limits["nvidia.com/gpu"] = str(config.gpu_count)

        # Pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "app": config.model_name,
                    "version": config.model_version,
                    "environment": config.environment,
                },
                annotations={
                    "prometheus.io/scrape": "true",
                    "prometheus.io/port": "9090",
                    "prometheus.io/path": "/metrics",
                },
            ),
            spec=client.V1PodSpec(
                containers=[container],
                volumes=[
                    client.V1Volume(
                        name="model-storage",
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=f"{config.model_name}-pvc"
                        ),
                    )
                ],
                image_pull_secrets=[
                    client.V1LocalObjectReference(name="registry-credentials")
                ]
                if config.environment == "production"
                else None,
                affinity=self._create_affinity(config)
                if config.environment == "production"
                else None,
                tolerations=self._create_tolerations(config)
                if config.gpu_enabled
                else None,
            ),
        )

        # Deployment spec
        spec = client.V1DeploymentSpec(
            replicas=config.replicas,
            selector=client.V1LabelSelector(
                match_labels={"app": config.model_name, "version": config.model_version}
            ),
            template=template,
            strategy=client.V1DeploymentStrategy(
                type="RollingUpdate",
                rolling_update=client.V1RollingUpdateDeployment(
                    max_unavailable="25%", max_surge="25%"
                ),
            ),
            revision_history_limit=5,
        )

        # Create deployment
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=f"{config.model_name}-deployment",
                namespace=config.namespace,
                labels={"app": config.model_name, "version": config.model_version},
            ),
            spec=spec,
        )

        try:
            return self.apps_v1.create_namespaced_deployment(
                namespace=config.namespace, body=deployment
            )
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                return self.apps_v1.patch_namespaced_deployment(
                    name=f"{config.model_name}-deployment",
                    namespace=config.namespace,
                    body=deployment,
                )
            raise

    def _create_canary_deployment(self, config: ModelDeploymentConfig, image_uri: str):
        """Create canary deployment."""
        # Create main deployment with reduced replicas
        main_config = config
        main_config.replicas = int(
            config.replicas * (100 - config.canary_percentage) / 100
        )
        main_deployment = self._create_deployment(main_config, image_uri)

        # Create canary deployment
        canary_config = config
        canary_config.model_version = f"{config.model_version}-canary"
        canary_config.replicas = int(config.replicas * config.canary_percentage / 100)
        canary_deployment = self._create_deployment(canary_config, image_uri)

        return canary_deployment

    def _create_service(self, config: ModelDeploymentConfig):
        """Create Kubernetes service."""
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=f"{config.model_name}-service",
                namespace=config.namespace,
                labels={"app": config.model_name},
            ),
            spec=client.V1ServiceSpec(
                selector={"app": config.model_name},
                ports=[
                    client.V1ServicePort(
                        name="http", port=80, target_port=8080, protocol="TCP"
                    ),
                    client.V1ServicePort(
                        name="metrics", port=9090, target_port=9090, protocol="TCP"
                    ),
                ],
                type="ClusterIP",
            ),
        )

        try:
            return self.v1.create_namespaced_service(
                namespace=config.namespace, body=service
            )
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                return self.v1.patch_namespaced_service(
                    name=f"{config.model_name}-service",
                    namespace=config.namespace,
                    body=service,
                )
            raise

    def _create_ingress(self, config: ModelDeploymentConfig):
        """Create Kubernetes ingress."""
        ingress = client.V1Ingress(
            api_version="networking.k8s.io/v1",
            kind="Ingress",
            metadata=client.V1ObjectMeta(
                name=f"{config.model_name}-ingress",
                namespace=config.namespace,
                annotations={
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/rate-limit": "100",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                },
            ),
            spec=client.V1IngressSpec(
                tls=[
                    client.V1IngressTLS(
                        hosts=[f"{config.model_name}.ml.example.com"],
                        secret_name=f"{config.model_name}-tls",
                    )
                ],
                rules=[
                    client.V1IngressRule(
                        host=f"{config.model_name}.ml.example.com",
                        http=client.V1HTTPIngressRuleValue(
                            paths=[
                                client.V1HTTPIngressPath(
                                    path="/",
                                    path_type="Prefix",
                                    backend=client.V1IngressBackend(
                                        service=client.V1IngressServiceBackend(
                                            name=f"{config.model_name}-service",
                                            port=client.V1ServiceBackendPort(number=80),
                                        )
                                    ),
                                )
                            ]
                        ),
                    )
                ],
            ),
        )

        try:
            return self.networking_v1.create_namespaced_ingress(
                namespace=config.namespace, body=ingress
            )
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                return self.networking_v1.patch_namespaced_ingress(
                    name=f"{config.model_name}-ingress",
                    namespace=config.namespace,
                    body=ingress,
                )
            raise

    def _create_hpa(self, config: ModelDeploymentConfig):
        """Create Horizontal Pod Autoscaler."""
        hpa = client.V1HorizontalPodAutoscaler(
            api_version="autoscaling/v1",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(
                name=f"{config.model_name}-hpa", namespace=config.namespace
            ),
            spec=client.V1HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V1CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=f"{config.model_name}-deployment",
                ),
                min_replicas=config.min_replicas,
                max_replicas=config.max_replicas,
                target_cpu_utilization_percentage=config.target_cpu_utilization,
            ),
        )

        try:
            return self.autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                namespace=config.namespace, body=hpa
            )
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                return self.autoscaling_v1.patch_namespaced_horizontal_pod_autoscaler(
                    name=f"{config.model_name}-hpa",
                    namespace=config.namespace,
                    body=hpa,
                )
            raise

    def _create_env_vars(self, config: ModelDeploymentConfig) -> list:
        """Create environment variables for container."""
        env_vars = [
            client.V1EnvVar(name="MODEL_NAME", value=config.model_name),
            client.V1EnvVar(name="MODEL_VERSION", value=config.model_version),
            client.V1EnvVar(name="MODEL_URI", value=config.model_uri),
            client.V1EnvVar(name="MODEL_TYPE", value=config.model_type),
            client.V1EnvVar(name="ENVIRONMENT", value=config.environment),
            client.V1EnvVar(name="CACHE_ENABLED", value=str(config.cache_enabled)),
            client.V1EnvVar(name="CACHE_TTL", value=str(config.cache_ttl)),
            client.V1EnvVar(
                name="MONITORING_ENABLED", value=str(config.monitoring_enabled)
            ),
            client.V1EnvVar(
                name="POD_NAME",
                value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(field_path="metadata.name")
                ),
            ),
            client.V1EnvVar(
                name="POD_NAMESPACE",
                value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(
                        field_path="metadata.namespace"
                    )
                ),
            ),
        ]

        # Add custom environment variables
        for key, value in config.environment_variables.items():
            env_vars.append(client.V1EnvVar(name=key, value=value))

        return env_vars

    def _create_affinity(self, config: ModelDeploymentConfig):
        """Create pod affinity rules."""
        return client.V1Affinity(
            pod_anti_affinity=client.V1PodAntiAffinity(
                preferred_during_scheduling_ignored_during_execution=[
                    client.V1WeightedPodAffinityTerm(
                        weight=100,
                        pod_affinity_term=client.V1PodAffinityTerm(
                            label_selector=client.V1LabelSelector(
                                match_labels={"app": config.model_name}
                            ),
                            topology_key="kubernetes.io/hostname",
                        ),
                    )
                ]
            )
        )

    def _create_tolerations(self, config: ModelDeploymentConfig):
        """Create GPU tolerations."""
        return [
            client.V1Toleration(
                key="nvidia.com/gpu", operator="Exists", effect="NoSchedule"
            )
        ]

    def _setup_monitoring(self, config: ModelDeploymentConfig) -> dict:
        """Setup monitoring for deployed model."""
        # Create ServiceMonitor for Prometheus
        service_monitor = {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": f"{config.model_name}-monitor",
                "namespace": config.namespace,
            },
            "spec": {
                "selector": {"matchLabels": {"app": config.model_name}},
                "endpoints": [
                    {"port": "metrics", "interval": "30s", "path": "/metrics"}
                ],
            },
        }

        # Create Grafana dashboard
        dashboard = self._create_grafana_dashboard(config)

        # Create alerts
        alerts = self._create_alerts(config)

        return {
            "service_monitor": service_monitor,
            "dashboard": dashboard,
            "alerts": alerts,
        }

    def _create_grafana_dashboard(self, config: ModelDeploymentConfig) -> dict:
        """Create Grafana dashboard for model monitoring."""
        dashboard = {
            "dashboard": {
                "title": f"{config.model_name} Monitoring",
                "panels": [
                    {
                        "title": "Prediction Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": f'rate(model_predictions_total{{model="{config.model_name}"}}[5m])'
                            }
                        ],
                    },
                    {
                        "title": "Prediction Latency",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": f'histogram_quantile(0.95, model_prediction_latency_seconds{{model="{config.model_name}"}})'
                            }
                        ],
                    },
                    {
                        "title": "Model Accuracy",
                        "type": "graph",
                        "targets": [
                            {"expr": f'model_accuracy{{model="{config.model_name}"}}'}
                        ],
                    },
                    {
                        "title": "Data Drift Score",
                        "type": "graph",
                        "targets": [
                            {"expr": f'data_drift_score{{model="{config.model_name}"}}'}
                        ],
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": f'rate(model_errors_total{{model="{config.model_name}"}}[5m])'
                            }
                        ],
                    },
                ],
            }
        }

        # Post to Grafana API
        # Implementation depends on your Grafana setup

        return dashboard

    def _create_alerts(self, config: ModelDeploymentConfig) -> list[dict]:
        """Create alert rules for model monitoring."""
        alerts = [
            {
                "alert": f"{config.model_name}_HighErrorRate",
                "expr": f'rate(model_errors_total{{model="{config.model_name}"}}[5m]) > 0.01',
                "for": "5m",
                "labels": {"severity": "warning", "model": config.model_name},
                "annotations": {
                    "summary": f"High error rate for {config.model_name}",
                    "description": "Error rate is above 1% for 5 minutes",
                },
            },
            {
                "alert": f"{config.model_name}_HighLatency",
                "expr": f'histogram_quantile(0.95, model_prediction_latency_seconds{{model="{config.model_name}"}}) > 1',
                "for": "5m",
                "labels": {"severity": "warning", "model": config.model_name},
                "annotations": {
                    "summary": f"High latency for {config.model_name}",
                    "description": "95th percentile latency is above 1 second",
                },
            },
            {
                "alert": f"{config.model_name}_DataDrift",
                "expr": f'data_drift_score{{model="{config.model_name}"}} > 0.1',
                "for": "10m",
                "labels": {"severity": "critical", "model": config.model_name},
                "annotations": {
                    "summary": f"Data drift detected for {config.model_name}",
                    "description": "Significant data drift detected, consider retraining",
                },
            },
        ]

        return alerts

    def _wait_for_deployment(
        self, config: ModelDeploymentConfig, timeout: int = 300
    ) -> bool:
        """Wait for deployment to be ready."""
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=f"{config.model_name}-deployment", namespace=config.namespace
                )

                if deployment.status.ready_replicas == config.replicas:
                    logger.info(f"Deployment {config.model_name} is ready")
                    return True

            except Exception as e:
                logger.error(f"Error checking deployment status: {e}")

            time.sleep(10)

        logger.warning(
            f"Deployment {config.model_name} not ready after {timeout} seconds"
        )
        return False

    def rollback_deployment(
        self, config: ModelDeploymentConfig, revision: int | None = None
    ):
        """Rollback deployment to previous version."""
        try:
            if revision is None:
                # Get previous revision
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=f"{config.model_name}-deployment", namespace=config.namespace
                )
                revision = deployment.metadata.annotations.get(
                    "deployment.kubernetes.io/revision", "1"
                )
                revision = str(int(revision) - 1)

            # Rollback using kubectl equivalent
            body = {
                "name": f"{config.model_name}-deployment",
                "rollout": {"revision": revision},
            }

            # Patch deployment to trigger rollback
            self.apps_v1.patch_namespaced_deployment(
                name=f"{config.model_name}-deployment",
                namespace=config.namespace,
                body=deployment,
            )

            logger.info(f"Rolled back {config.model_name} to revision {revision}")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise

    def _fetch_secret_from_vault(self, secret_name: str) -> dict[str, str]:
        """Fetch secret from HashiCorp Vault or AWS Secrets Manager."""
        # Example for AWS Secrets Manager
        client = boto3.client("secretsmanager")

        try:
            response = client.get_secret_value(SecretId=secret_name)
            return json.loads(response["SecretString"])
        except Exception as e:
            logger.error(f"Failed to fetch secret {secret_name}: {e}")
            return {}

    def _prepare_model_serving_code(
        self, config: ModelDeploymentConfig, build_context: Path
    ):
        """Prepare model serving code for Docker build."""
        # Create requirements.txt
        requirements = [
            "fastapi==0.68.0",
            "uvicorn==0.15.0",
            "prometheus-client==0.11.0",
            "redis==3.5.3",
            "mlflow==1.20.0",
            "numpy==1.21.0",
            "pandas==1.3.0",
            "scikit-learn==0.24.2",
            "pydantic==1.8.2",
        ]

        if config.model_type == "tensorflow":
            requirements.append("tensorflow==2.6.0")
        elif config.model_type == "pytorch":
            requirements.append("torch==1.9.0")
        elif config.model_type == "xgboost":
            requirements.append("xgboost==1.4.0")

        with open(build_context / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))

        # Copy model server code (should be in a template directory)
        # This is a simplified version - in production, copy from templates
        with open(build_context / "model_server.py", "w") as f:
            f.write("""
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app
import mlflow
import numpy as np
import pandas as pd
import os
import redis
import json
import hashlib

app = FastAPI()

# Load model
model = mlflow.pyfunc.load_model(os.environ['MODEL_URI'])

# Redis cache
cache = redis.Redis(host='redis', port=6379, db=0)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}

@app.post("/predict")
async def predict(data: dict):
    try:
        # Check cache
        cache_key = hashlib.md5(json.dumps(data).encode()).hexdigest()
        cached = cache.get(cache_key)
        if cached:
            return json.loads(cached)

        # Predict
        df = pd.DataFrame([data])
        prediction = model.predict(df)

        result = {"prediction": prediction.tolist()}

        # Cache result
        cache.setex(cache_key, 300, json.dumps(result))

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
""")


class ModelMonitor:
    """Monitor deployed models."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis.Redis(host="redis", port=6379, db=1)

    async def monitor_predictions(
        self,
        model_name: str,
        model_version: str,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """Monitor predictions and update metrics."""
        try:
            # Update counters
            prediction_counter.labels(
                model=model_name, version=model_version, status="success"
            ).inc(len(y_pred))

            # Calculate and update accuracy
            accuracy = (y_true == y_pred).mean()
            model_accuracy.labels(model=model_name, version=model_version).set(accuracy)

            # Check for data drift
            drift_scores = await self._calculate_drift(model_name, X)
            for feature, score in drift_scores.items():
                data_drift_score.labels(model=model_name, feature=feature).set(score)

            # Store metrics in Redis for historical tracking
            timestamp = datetime.now().isoformat()
            metrics_data = {
                "timestamp": timestamp,
                "accuracy": accuracy,
                "drift_scores": drift_scores,
                "n_predictions": len(y_pred),
            }

            self.redis_client.zadd(
                f"metrics:{model_name}:{model_version}",
                {json.dumps(metrics_data): datetime.now().timestamp()},
            )

            # Check if retraining is needed
            if accuracy < 0.8 or any(score > 0.15 for score in drift_scores.values()):
                await self._trigger_retraining(
                    model_name, model_version, accuracy, drift_scores
                )

        except Exception as e:
            self.logger.error(f"Error monitoring predictions: {e}")
            error_counter.labels(
                model=model_name, version=model_version, error_type="monitoring"
            ).inc()

    async def _calculate_drift(
        self, model_name: str, X: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate data drift using various methods."""
        drift_scores = {}

        # Load reference distribution from Redis
        reference_key = f"reference:{model_name}"
        reference_data = self.redis_client.get(reference_key)

        if not reference_data:
            # Store current as reference if not exists
            reference_stats = {}
            for column in X.columns:
                reference_stats[column] = {
                    "mean": float(X[column].mean()),
                    "std": float(X[column].std()),
                    "min": float(X[column].min()),
                    "max": float(X[column].max()),
                }
            self.redis_client.set(reference_key, json.dumps(reference_stats))
            return {col: 0.0 for col in X.columns}

        reference_stats = json.loads(reference_data)

        for column in X.columns:
            if column not in reference_stats:
                continue

            current_stats = {
                "mean": float(X[column].mean()),
                "std": float(X[column].std()),
                "min": float(X[column].min()),
                "max": float(X[column].max()),
            }

            # Calculate drift score (simplified KL divergence)
            drift_score = self._calculate_kl_divergence(
                reference_stats[column], current_stats
            )

            drift_scores[column] = drift_score

        return drift_scores

    def _calculate_kl_divergence(self, ref_stats: dict, curr_stats: dict) -> float:
        """Calculate simplified KL divergence between distributions."""
        # Simplified calculation using normal approximation
        ref_mean = ref_stats["mean"]
        ref_std = max(ref_stats["std"], 0.001)
        curr_mean = curr_stats["mean"]
        curr_std = max(curr_stats["std"], 0.001)

        # KL divergence for normal distributions
        kl = (
            np.log(curr_std / ref_std)
            + (ref_std**2 + (ref_mean - curr_mean) ** 2) / (2 * curr_std**2)
            - 0.5
        )

        return float(kl)

    async def _trigger_retraining(
        self,
        model_name: str,
        model_version: str,
        accuracy: float,
        drift_scores: dict[str, float],
    ):
        """Trigger model retraining pipeline."""
        self.logger.warning(f"Triggering retraining for {model_name} v{model_version}")
        self.logger.warning(
            f"Reason: Accuracy={accuracy:.3f}, Max drift={max(drift_scores.values()):.3f}"
        )

        # Trigger Airflow DAG
        async with aiohttp.ClientSession() as session:
            payload = {
                "conf": {
                    "model_name": model_name,
                    "model_version": model_version,
                    "trigger_reason": "performance_degradation",
                    "accuracy": accuracy,
                    "drift_scores": drift_scores,
                }
            }

            try:
                async with session.post(
                    "http://airflow-webserver:8080/api/v1/dags/retrain_model/dagRuns",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        self.logger.info(
                            f"Retraining triggered successfully for {model_name}"
                        )
                    else:
                        self.logger.error(
                            f"Failed to trigger retraining: {response.status}"
                        )
            except Exception as e:
                self.logger.error(f"Error triggering retraining: {e}")


def deploy_model_from_mlflow(
    model_name: str, model_version: str, environment: str = "staging"
) -> dict:
    """Deploy model from MLflow Model Registry."""
    # Get model URI from MLflow
    mlflow_client = mlflow.tracking.MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"

    # Get model metadata
    model_details = mlflow_client.get_model_version(model_name, model_version)

    # Create deployment configuration
    config = ModelDeploymentConfig(
        model_name=model_name,
        model_version=model_version,
        model_uri=model_uri,
        model_type=model_details.tags.get("model_type", "sklearn"),
        environment=environment,
        replicas=2 if environment == "staging" else 4,
        autoscaling=environment == "production",
        monitoring_enabled=True,
        cache_enabled=True,
    )

    # Deploy
    manager = ModelDeploymentManager()
    result = manager.deploy_model(config)

    return result
