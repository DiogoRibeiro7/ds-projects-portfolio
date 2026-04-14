# ML API Inference Service Usage

This page shows example client usage for the ML inference service exposed by `src/api/ml_api.py`.

## Start the service

Run the FastAPI application locally:

```bash
uvicorn src.api.ml_api:app --reload --host 0.0.0.0 --port 8000
```

## Health check

Check whether the service is available:

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "timestamp": "2026-04-14T12:00:00",
  "models": {},
  "cache_status": "unavailable",
  "database_status": "healthy"
}
```

## Single prediction

Send a single prediction request to `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature_1": 5.2,
      "feature_2": 1.3,
      "feature_3": "A"
    },
    "model_name": "default",
    "model_version": "latest",
    "explain": false
  }'
```

Example response:

```json
{
  "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
  "prediction": 1,
  "probability": [0.2, 0.8],
  "model_name": "default",
  "model_version": "latest",
  "timestamp": "2026-04-14T12:01:00",
  "latency_ms": 45.1,
  "explanation": null
}
```

## Batch prediction

Send multiple instances to `/batch_predict`:

```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"feature_1": 5.2, "feature_2": 1.3, "feature_3": "A"},
      {"feature_1": 6.1, "feature_2": 0.8, "feature_3": "B"}
    ],
    "model_name": "default",
    "model_version": "latest",
    "batch_size": 100
  }'
```

Example response:

```json
{
  "predictions": [
    {"prediction": 1, "probability": [0.2, 0.8]},
    {"prediction": 0, "probability": [0.7, 0.3]}
  ],
  "count": 2,
  "model_name": "default",
  "model_version": "latest",
  "latency_ms": 120.5
}
```

## Python client example

A reusable example client is available at `examples/inference_service_api_usage.py`.

```bash
python examples/inference_service_api_usage.py
```

## Notes

- The service also exposes `/metrics` for Prometheus and `/ready` for readiness checks.
- Use `model_name` and `model_version` to target specific model deployments.
