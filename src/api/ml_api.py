"""ML API Service
Production-ready FastAPI service for model serving and inference
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import aioredis
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import prometheus_client
import shap
import uvloop
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set event loop policy for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Prometheus metrics
request_count = Counter(
    "ml_api_requests_total", "Total API requests", ["endpoint", "status"]
)
request_latency = Histogram(
    "ml_api_request_latency_seconds", "Request latency", ["endpoint"]
)
prediction_count = Counter(
    "ml_api_predictions_total", "Total predictions made", ["model"]
)
model_version_gauge = Gauge(
    "ml_api_model_version", "Current model version", ["model_name"]
)
active_models = Gauge("ml_api_active_models", "Number of active models")


# Request/Response models
class PredictionRequest(BaseModel):
    """Single prediction request"""

    features: dict[str, float | int | str]
    model_name: str | None = "default"
    model_version: str | None = "latest"
    explain: bool | None = False

    @validator("features")
    def validate_features(cls, v):
        """Validate that a prediction request contains at least one feature.

        Args:
            v: Mapping of feature names to scalar inputs.

        Returns:
            The original feature mapping when it is non-empty.

        Raises:
            ValueError: If no features were provided.
        """
        if not v:
            raise ValueError("Features cannot be empty")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""

    instances: list[dict[str, float | int | str]]
    model_name: str | None = "default"
    model_version: str | None = "latest"
    batch_size: int | None = Field(100, le=10000)

    @validator("instances")
    def validate_instances(cls, v):
        """Ensure batch requests contain data but stay within safety limits.

        Args:
            v: Sequence of JSON-serializable feature dictionaries.

        Returns:
            The original list when it contains at least one instance and no more
            than 10,000 rows.

        Raises:
            ValueError: If the batch is empty or exceeds the hard limit.
        """
        if not v:
            raise ValueError("Instances cannot be empty")
        if len(v) > 10000:
            raise ValueError("Batch size cannot exceed 10000")
        return v


class PredictionResponse(BaseModel):
    """Prediction response"""

    prediction_id: str
    prediction: float | int | str | list
    probability: list[float] | None = None
    model_name: str
    model_version: str
    timestamp: datetime
    latency_ms: float
    explanation: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: datetime
    models: dict[str, dict[str, Any]]
    cache_status: str
    database_status: str


class ModelManager:
    """Manages model loading, caching, and serving"""

    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.feature_names = {}
        self.feature_transformers = {}
        self.explainers = {}

    async def load_model(self, model_name: str, model_version: str = "latest"):
        """Load model from MLflow or file system"""
        try:
            model_key = f"{model_name}:{model_version}"

            # Check if model already loaded
            if model_key in self.models:
                return self.models[model_key]

            # Try loading from MLflow
            try:
                client = mlflow.tracking.MlflowClient()
                model_uri = f"models:/{model_name}/{model_version}"
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Loaded model {model_name}:{model_version} from MLflow")
            except Exception as e:
                logger.warning(f"Could not load from MLflow: {e}, trying file system")
                # Fallback to file system
                model_path = Path(f"/app/models/{model_name}/{model_version}/model.pkl")
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                else:
                    raise FileNotFoundError(
                        f"Model {model_name}:{model_version} not found"
                    )

            # Store model and metadata
            self.models[model_key] = model
            self.model_metadata[model_key] = {
                "loaded_at": datetime.now(),
                "version": model_version,
                "name": model_name,
            }

            # Update metrics
            active_models.set(len(self.models))
            model_version_gauge.labels(model_name=model_name).set(
                float(
                    model_version.replace("v", "") if model_version != "latest" else 0
                )
            )

            # Initialize SHAP explainer if model supports it
            try:
                if hasattr(model, "predict"):
                    self.explainers[model_key] = shap.Explainer(model)
                    logger.info(f"Initialized SHAP explainer for {model_key}")
            except Exception as e:
                logger.warning(f"Could not initialize explainer: {e}")

            return model

        except Exception as e:
            logger.error(f"Error loading model {model_name}:{model_version}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def predict(self, features: dict, model_name: str, model_version: str):
        """Make prediction with the specified model"""
        model_key = f"{model_name}:{model_version}"
        model = await self.load_model(model_name, model_version)

        # Convert features to DataFrame
        df = pd.DataFrame([features])

        # Make prediction
        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(df)
                prediction = np.argmax(probabilities, axis=1)[0]
                probability = probabilities[0].tolist()
            else:
                prediction = model.predict(df)[0]
                probability = None

            return prediction, probability

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def explain_prediction(
        self, features: dict, model_name: str, model_version: str
    ):
        """Generate explanation for prediction"""
        model_key = f"{model_name}:{model_version}"

        if model_key not in self.explainers:
            return None

        try:
            df = pd.DataFrame([features])
            explainer = self.explainers[model_key]
            shap_values = explainer(df)

            explanation = {
                "shap_values": shap_values.values[0].tolist(),
                "base_value": float(shap_values.base_values[0]),
                "feature_names": list(features.keys()),
                "feature_importance": dict(
                    zip(
                        features.keys(),
                        np.abs(shap_values.values[0]).tolist(),
                        strict=False,
                    )
                ),
            }

            return explanation

        except Exception as e:
            logger.warning(f"Could not generate explanation: {e}")
            return None


class CacheManager:
    """Manages Redis cache for predictions"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None

    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await aioredis.create_redis_pool(
                self.redis_url, minsize=5, maxsize=20
            )
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None

    async def get(self, key: str):
        """Get value from cache"""
        if not self.redis:
            return None

        try:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        if not self.redis:
            return

        try:
            await self.redis.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    async def close(self):
        """Close Redis connection"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()

    def generate_cache_key(self, features: dict, model_name: str, model_version: str):
        """Generate cache key for prediction"""
        feature_str = json.dumps(features, sort_keys=True)
        key_str = f"{model_name}:{model_version}:{feature_str}"
        return hashlib.md5(key_str.encode()).hexdigest()


# Global instances
model_manager = ModelManager()
cache_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    global cache_manager
    redis_url = "redis://redis-service:6379"
    cache_manager = CacheManager(redis_url)
    await cache_manager.connect()

    # Load default models
    try:
        await model_manager.load_model("default", "latest")
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")

    yield

    # Shutdown
    await cache_manager.close()


# Create FastAPI app
app = FastAPI(
    title="ML Portfolio API",
    version="1.0.0",
    description="Production-ready ML model serving API",
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
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "ML Portfolio API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "metrics": "/metrics",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check cache status
    cache_status = "healthy" if cache_manager and cache_manager.redis else "unavailable"

    # Check database status (placeholder)
    database_status = "healthy"

    # Get model information
    models = {}
    for model_key, metadata in model_manager.model_metadata.items():
        models[model_key] = {
            "loaded_at": metadata["loaded_at"].isoformat(),
            "version": metadata["version"],
        }

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models=models,
        cache_status=cache_status,
        database_status=database_status,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Single prediction endpoint"""
    start_time = time.time()
    prediction_id = str(uuid.uuid4())

    # Check cache
    cache_key = cache_manager.generate_cache_key(
        request.features, request.model_name, request.model_version
    )
    cached_result = await cache_manager.get(cache_key)

    if cached_result:
        logger.info(f"Cache hit for prediction {prediction_id}")
        return PredictionResponse(**cached_result, prediction_id=prediction_id)

    # Make prediction
    try:
        prediction, probability = await model_manager.predict(
            request.features, request.model_name, request.model_version
        )

        # Generate explanation if requested
        explanation = None
        if request.explain:
            explanation = await model_manager.explain_prediction(
                request.features, request.model_name, request.model_version
            )

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Create response
        response_data = {
            "prediction": prediction,
            "probability": probability,
            "model_name": request.model_name,
            "model_version": request.model_version,
            "timestamp": datetime.now(),
            "latency_ms": latency_ms,
            "explanation": explanation,
        }

        # Cache result
        background_tasks.add_task(cache_manager.set, cache_key, response_data, 3600)

        # Update metrics
        prediction_count.labels(model=request.model_name).inc()
        request_latency.labels(endpoint="predict").observe(latency_ms / 1000)
        request_count.labels(endpoint="predict", status="success").inc()

        return PredictionResponse(prediction_id=prediction_id, **response_data)

    except Exception as e:
        request_count.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    start_time = time.time()
    results = []

    try:
        # Process in batches
        for i in range(0, len(request.instances), request.batch_size):
            batch = request.instances[i : i + request.batch_size]

            # Make predictions for batch
            batch_predictions = []
            for instance in batch:
                prediction, probability = await model_manager.predict(
                    instance, request.model_name, request.model_version
                )
                batch_predictions.append(
                    {"prediction": prediction, "probability": probability}
                )

            results.extend(batch_predictions)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Update metrics
        prediction_count.labels(model=request.model_name).inc(len(request.instances))
        request_latency.labels(endpoint="batch_predict").observe(latency_ms / 1000)
        request_count.labels(endpoint="batch_predict", status="success").inc()

        return {
            "predictions": results,
            "count": len(results),
            "model_name": request.model_name,
            "model_version": request.model_version,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        request_count.labels(endpoint="batch_predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=prometheus_client.generate_latest(), media_type="text/plain"
    )


@app.get("/ready")
async def readiness():
    """Readiness probe"""
    # Check if at least one model is loaded
    if not model_manager.models:
        raise HTTPException(status_code=503, detail="No models loaded")
    return {"ready": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
