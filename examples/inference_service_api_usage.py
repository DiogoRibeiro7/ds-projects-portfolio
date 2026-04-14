"""Example client usage for the ML inference service.

This script demonstrates how to call the FastAPI inference endpoints
exposed by `src/api/ml_api.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

API_BASE_URL = "http://localhost:8000"


def health_check() -> None:
    url = f"{API_BASE_URL}/health"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    print("Health check response:")
    print(json.dumps(response.json(), indent=2))


def single_prediction() -> None:
    url = f"{API_BASE_URL}/predict"
    payload = {
        "features": {
            "feature_1": 5.2,
            "feature_2": 1.3,
            "feature_3": "A",
        },
        "model_name": "default",
        "model_version": "latest",
        "explain": False,
    }
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    print("Single prediction response:")
    print(json.dumps(response.json(), indent=2))


def batch_prediction() -> None:
    url = f"{API_BASE_URL}/batch_predict"
    payload = {
        "instances": [
            {"feature_1": 5.2, "feature_2": 1.3, "feature_3": "A"},
            {"feature_1": 6.1, "feature_2": 0.8, "feature_3": "B"},
        ],
        "model_name": "default",
        "model_version": "latest",
        "batch_size": 100,
    }
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    print("Batch prediction response:")
    print(json.dumps(response.json(), indent=2))


def main() -> None:
    print("Running inference service API examples...\n")
    health_check()
    print("\n")
    single_prediction()
    print("\n")
    batch_prediction()


if __name__ == "__main__":
    main()
