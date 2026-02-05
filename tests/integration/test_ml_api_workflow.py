"""
Integration tests for the FastAPI prediction workflow.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

# Ensure aioredis dependency is satisfied without pulling the real package.
import sys
import types

if "aioredis" not in sys.modules:
    aioredis_stub = types.ModuleType("aioredis")

    async def _missing_pool(*args, **kwargs):
        raise RuntimeError("aioredis not available in integration tests")

    aioredis_stub.create_redis_pool = _missing_pool
    sys.modules["aioredis"] = aioredis_stub

if "uvloop" not in sys.modules:
    uvloop_stub = types.ModuleType("uvloop")

    class _FakePolicy(asyncio.DefaultEventLoopPolicy):
        pass

    uvloop_stub.EventLoopPolicy = _FakePolicy
    sys.modules["uvloop"] = uvloop_stub

from src.api import ml_api

pytestmark = pytest.mark.integration


class StubModelManager:
    """Simple in-memory model registry used for tests."""

    def __init__(self):
        now = datetime.now()
        self.model_metadata: Dict[str, Dict[str, Any]] = {
            "default:latest": {"loaded_at": now, "version": "latest"}
        }
        self.models = {"default:latest": object()}
        self.predict_calls = 0

    async def load_model(self, model_name: str, model_version: str):
        key = f"{model_name}:{model_version}"
        self.models[key] = object()
        self.model_metadata[key] = {
            "loaded_at": datetime.now(),
            "version": model_version,
        }

    async def predict(
        self, features: Dict[str, Any], model_name: str, model_version: str
    ) -> Tuple[int, List[float]]:
        self.predict_calls += 1
        return 1, [0.25, 0.75]

    async def explain_prediction(
        self, features: Dict[str, Any], model_name: str, model_version: str
    ) -> Dict[str, Any]:
        return {
            "shap_values": [0.1 for _ in features],
            "base_value": 0.0,
            "feature_names": list(features.keys()),
            "feature_importance": {k: 0.1 for k in features},
        }


class StubCacheManager:
    """Async-compatible cache facade used by the FastAPI app."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = True  # health flag
        self.store: Dict[str, Dict[str, Any]] = {}

    async def connect(self):
        return None

    async def close(self):
        return None

    async def get(self, key: str):
        await asyncio.sleep(0)
        return self.store.get(key)

    def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        self.store[key] = value

    def generate_cache_key(
        self, features: Dict[str, Any], model_name: str, model_version: str
    ):
        return f"{model_name}:{model_version}:{tuple(sorted(features.items()))}"


@pytest.fixture()
def api_client(monkeypatch):
    """Provide a TestClient wired to stub managers."""
    stub_model_manager = StubModelManager()
    monkeypatch.setattr(ml_api, "model_manager", stub_model_manager)
    monkeypatch.setattr(ml_api, "CacheManager", StubCacheManager)
    with TestClient(ml_api.app) as client:
        yield client, stub_model_manager, ml_api.cache_manager


def test_predict_endpoint_caches_results(api_client):
    client, stub_model_manager, cache_manager = api_client

    payload = {
        "features": {"CreditScore": 700, "Age": 35, "Balance": 50000.0},
        "model_name": "default",
        "model_version": "latest",
        "explain": False,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] == 1
    assert body["probability"] == [0.25, 0.75]
    assert body["model_name"] == "default"
    assert "prediction_id" in body
    assert stub_model_manager.predict_calls == 1

    # Cached response should skip model invocation
    cached_response = client.post("/predict", json=payload)
    assert cached_response.status_code == 200
    assert stub_model_manager.predict_calls == 1
    assert cache_manager.store  # ensure the cache was written
