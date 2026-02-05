"""
Production Model Server with FastAPI
=====================================

A high-performance, production-ready model serving application with:
- Async request handling
- Model versioning and A/B testing
- Distributed tracing
- Prometheus metrics
- Request/response validation
- Caching
- Circuit breaker pattern
- Health checks
"""

import os
import time
import json
import asyncio
import logging
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from enum import Enum

import numpy as np
import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
    Depends,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# ML Libraries
import joblib
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator

# Monitoring and tracing
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter

# Caching
import redis
import aiocache
from aiocache.serializers import JsonSerializer

# Utilities
import structlog
from python_json_logger import JsonFormatter

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
prediction_counter = Counter(
    "model_predictions_total",
    "Total number of predictions",
    ["model_name", "version", "status"],
)
prediction_latency = Histogram(
    "model_prediction_duration_seconds", "Prediction latency", ["model_name", "version"]
)
model_load_time = Gauge(
    "model_load_time_seconds", "Time taken to load model", ["model_name", "version"]
)
active_models = Gauge("active_models_count", "Number of active models")
cache_hits = Counter("cache_hits_total", "Total cache hits")
cache_misses = Counter("cache_misses_total", "Total cache misses")

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
    agent_port=int(os.getenv("JAEGER_AGENT_PORT", 6831)),
)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)


# Request/Response models
class PredictionRequest(BaseModel):
    """Model prediction request schema"""

    data: Union[List[List[float]], Dict[str, List[float]]]
    model_name: str = Field(..., description="Name of the model to use")
    model_version: Optional[str] = Field("latest", description="Version of the model")
    features: Optional[List[str]] = Field(
        None, description="Feature names if data is list format"
    )
    request_id: Optional[str] = Field(None, description="Unique request ID for tracing")
    timeout: Optional[int] = Field(30, description="Request timeout in seconds")

    @validator("data")
    def validate_data(cls, v):
        if isinstance(v, list):
            if not all(isinstance(row, list) for row in v):
                raise ValueError("All rows must be lists")
        return v


class PredictionResponse(BaseModel):
    """Model prediction response schema"""

    predictions: List[Union[float, int, str, Dict]]
    model_name: str
    model_version: str
    request_id: str
    timestamp: str
    latency_ms: float
    cached: bool = False
    metadata: Optional[Dict] = None


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ModelInfo(BaseModel):
    """Model information schema"""

    name: str
    version: str
    framework: str
    loaded_at: str
    metrics: Dict[str, float]
    status: str


# Model Manager
class ModelManager:
    """Manages model loading, caching, and versioning"""

    def __init__(self):
        self.models: Dict[str, Dict[str, BaseEstimator]] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.redis_client = None
        self.cache = aiocache.Cache(aiocache.SimpleMemoryCache)
        self.cache.serializer = JsonSerializer()
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection for distributed caching"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD"),
                decode_responses=True,
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using local cache only.")
            self.redis_client = None

    async def load_model(
        self, model_name: str, model_version: str = "latest"
    ) -> BaseEstimator:
        """Load model from MLflow or local storage"""
        start_time = time.time()
        model_key = f"{model_name}:{model_version}"

        # Check if model is already loaded
        if model_name in self.models and model_version in self.models[model_name]:
            logger.info(f"Model {model_key} already loaded")
            return self.models[model_name][model_version]

        try:
            # Try loading from MLflow
            if os.getenv("MLFLOW_TRACKING_URI"):
                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
                if model_version == "latest":
                    model_version = self._get_latest_version(model_name)

                model_uri = f"models:/{model_name}/{model_version}"
                model = mlflow.sklearn.load_model(model_uri)
            else:
                # Load from local file system
                model_path = f"/app/models/{model_name}/{model_version}/model.pkl"
                model = joblib.load(model_path)

            # Store model
            if model_name not in self.models:
                self.models[model_name] = {}
            self.models[model_name][model_version] = model

            # Store metadata
            self.model_metadata[model_key] = {
                "loaded_at": datetime.utcnow().isoformat(),
                "framework": type(model).__module__.split(".")[0],
                "version": model_version,
            }

            load_time = time.time() - start_time
            model_load_time.labels(model_name=model_name, version=model_version).set(
                load_time
            )
            active_models.inc()

            logger.info(f"Model {model_key} loaded successfully", load_time=load_time)
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_key}", error=str(e))
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )

    def _get_latest_version(self, model_name: str) -> str:
        """Get latest model version from MLflow"""
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")
        return max(versions, key=lambda x: int(x.version)).version

    async def predict(
        self, model_name: str, model_version: str, data: np.ndarray
    ) -> np.ndarray:
        """Make prediction using specified model"""
        model = await self.load_model(model_name, model_version)

        with tracer.start_as_current_span("model_prediction"):
            predictions = model.predict(data)

        return predictions

    def get_model_info(self, model_name: str, model_version: str) -> Dict:
        """Get information about a loaded model"""
        model_key = f"{model_name}:{model_version}"
        if model_key not in self.model_metadata:
            return None
        return self.model_metadata[model_key]


# Circuit Breaker
class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if (
                datetime.utcnow() - self.last_failure_time
            ).seconds > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise HTTPException(
                    status_code=503, detail="Service temporarily unavailable"
                )

        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error("Circuit breaker opened", failures=self.failure_count)

            raise e


# Initialize components
model_manager = ModelManager()
circuit_breaker = CircuitBreaker()


# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting model server...")

    # Pre-load default models if specified
    default_models = os.getenv("PRELOAD_MODELS", "").split(",")
    for model_spec in default_models:
        if model_spec:
            parts = model_spec.split(":")
            model_name = parts[0]
            model_version = parts[1] if len(parts) > 1 else "latest"
            try:
                await model_manager.load_model(model_name, model_version)
            except Exception as e:
                logger.error(f"Failed to preload model {model_spec}", error=str(e))

    yield

    # Shutdown
    logger.info("Shutting down model server...")
    if model_manager.redis_client:
        model_manager.redis_client.close()


app = FastAPI(
    title="ML Model Server",
    description="Production-ready ML model serving with monitoring and tracing",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID for tracing"""
    request_id = request.headers.get("X-Request-ID", str(datetime.utcnow().timestamp()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_healthy = True
        if model_manager.redis_client:
            try:
                model_manager.redis_client.ping()
            except:
                redis_healthy = False

        # Check loaded models
        models_count = sum(len(versions) for versions in model_manager.models.values())

        status = HealthStatus.HEALTHY
        if not redis_healthy:
            status = HealthStatus.DEGRADED
        if models_count == 0:
            status = HealthStatus.UNHEALTHY

        return {
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "models_loaded": models_count,
            "redis_connected": redis_healthy,
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": HealthStatus.UNHEALTHY.value, "error": str(e)},
        )


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    models_count = sum(len(versions) for versions in model_manager.models.values())
    if models_count > 0 or not os.getenv("PRELOAD_MODELS"):
        return {"ready": True}
    return JSONResponse(status_code=503, content={"ready": False})


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Main prediction endpoint"""
    start_time = time.time()
    request_id = request.request_id or str(datetime.utcnow().timestamp())

    with tracer.start_as_current_span("prediction_request") as span:
        span.set_attribute("model.name", request.model_name)
        span.set_attribute("model.version", request.model_version)
        span.set_attribute("request.id", request_id)

        try:
            # Convert data to numpy array
            if isinstance(request.data, dict):
                df = pd.DataFrame(request.data)
                data_array = df.values
            else:
                data_array = np.array(request.data)

            # Check cache
            cache_key = hashlib.md5(
                f"{request.model_name}:{request.model_version}:{str(data_array)}".encode()
            ).hexdigest()

            cached_result = await model_manager.cache.get(cache_key)
            if cached_result:
                cache_hits.inc()
                logger.info("Cache hit", request_id=request_id)
                return PredictionResponse(
                    predictions=cached_result,
                    model_name=request.model_name,
                    model_version=request.model_version,
                    request_id=request_id,
                    timestamp=datetime.utcnow().isoformat(),
                    latency_ms=(time.time() - start_time) * 1000,
                    cached=True,
                )

            cache_misses.inc()

            # Make prediction with circuit breaker
            predictions = await circuit_breaker.call(
                model_manager.predict,
                request.model_name,
                request.model_version,
                data_array,
            )

            # Convert predictions to list
            if hasattr(predictions, "tolist"):
                predictions_list = predictions.tolist()
            else:
                predictions_list = list(predictions)

            # Cache result
            await model_manager.cache.set(cache_key, predictions_list, ttl=300)

            # Update metrics
            latency = time.time() - start_time
            prediction_counter.labels(
                model_name=request.model_name,
                version=request.model_version,
                status="success",
            ).inc()
            prediction_latency.labels(
                model_name=request.model_name, version=request.model_version
            ).observe(latency)

            # Log prediction
            logger.info(
                "Prediction completed",
                request_id=request_id,
                model=request.model_name,
                version=request.model_version,
                latency=latency,
                input_shape=data_array.shape,
                output_shape=len(predictions_list),
            )

            return PredictionResponse(
                predictions=predictions_list,
                model_name=request.model_name,
                model_version=request.model_version,
                request_id=request_id,
                timestamp=datetime.utcnow().isoformat(),
                latency_ms=latency * 1000,
                cached=False,
                metadata=model_manager.get_model_info(
                    request.model_name, request.model_version
                ),
            )

        except Exception as e:
            prediction_counter.labels(
                model_name=request.model_name,
                version=request.model_version,
                status="error",
            ).inc()
            logger.error(
                "Prediction failed",
                request_id=request_id,
                model=request.model_name,
                error=str(e),
            )
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    tasks = [predict(req, BackgroundTasks()) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            responses.append({"error": str(result), "request_index": i})
        else:
            responses.append(result)

    return responses


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all loaded models"""
    models_info = []
    for model_name, versions in model_manager.models.items():
        for version, model in versions.items():
            model_key = f"{model_name}:{version}"
            metadata = model_manager.model_metadata.get(model_key, {})
            models_info.append(
                ModelInfo(
                    name=model_name,
                    version=version,
                    framework=metadata.get("framework", "unknown"),
                    loaded_at=metadata.get("loaded_at", ""),
                    metrics={
                        "predictions_total": 0,  # Would need to track per model
                        "avg_latency_ms": 0,
                    },
                    status="active",
                )
            )
    return models_info


@app.post("/models/{model_name}/load")
async def load_model(model_name: str, version: str = "latest"):
    """Load a specific model version"""
    try:
        await model_manager.load_model(model_name, version)
        return {"message": f"Model {model_name}:{version} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/models/{model_name}/{version}")
async def unload_model(model_name: str, version: str):
    """Unload a specific model version"""
    if (
        model_name in model_manager.models
        and version in model_manager.models[model_name]
    ):
        del model_manager.models[model_name][version]
        if not model_manager.models[model_name]:
            del model_manager.models[model_name]
        active_models.dec()
        return {"message": f"Model {model_name}:{version} unloaded successfully"}
    raise HTTPException(status_code=404, detail="Model not found")


@app.post("/feedback")
async def submit_feedback(
    request_id: str,
    actual_value: Union[float, int, str],
    model_name: str,
    model_version: str,
):
    """Submit feedback for model predictions"""
    # Store feedback for model retraining
    feedback_data = {
        "request_id": request_id,
        "actual_value": actual_value,
        "model_name": model_name,
        "model_version": model_version,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Store in Redis or database
    if model_manager.redis_client:
        model_manager.redis_client.lpush(
            f"feedback:{model_name}:{model_version}", json.dumps(feedback_data)
        )

    logger.info("Feedback received", **feedback_data)
    return {"message": "Feedback recorded successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
