"""UN Comtrade API ingestion and provenance utilities.

The semiconductor case study uses official UN Comtrade endpoints only.  This
module keeps HTTP transport, canonical response hashing, and schema extraction
separate from substantive trade metrics so an API/schema change fails before it
can silently alter the empirical estimand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import time
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from curl_cffi import requests

RETRY_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


@dataclass(frozen=True)
class ComtradeJSONResponse:
    """One official Comtrade JSON response plus reproducibility metadata."""

    endpoint: str
    query: dict[str, str]
    retrieved_at_utc: str
    canonical_sha256: str
    payload: object


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize parsed JSON deterministically for a stable evidence digest."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    """Return the SHA-256 of the canonical JSON representation."""
    return sha256(canonical_json_bytes(payload)).hexdigest()


def redact_subscription_key(url: str) -> str:
    """Remove any Comtrade subscription key from a URL before recording it."""
    parts = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() != "subscription-key"
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def extract_data_rows(payload: object) -> list[dict[str, Any]]:
    """Extract the row list from a standard Comtrade JSON response."""
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    else:
        raise ValueError("UN Comtrade response does not contain a JSON data row list.")

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("UN Comtrade data rows must be JSON objects.")
    return [dict(row) for row in rows]


def response_schema(rows: list[dict[str, Any]]) -> dict[str, object]:
    """Return a compact field/type schema without interpreting trade values."""
    fields = sorted({str(key) for row in rows for key in row})
    types: dict[str, list[str]] = {}
    for field in fields:
        observed = sorted({type(row[field]).__name__ for row in rows if field in row})
        types[field] = observed
    return {"row_count": len(rows), "fields": fields, "field_types": types}


def get_official_json(
    endpoint: str,
    params: Mapping[str, object | None],
    *,
    subscription_key: str | None = None,
) -> ComtradeJSONResponse:
    """GET one official Comtrade JSON endpoint with bounded retries.

    The subscription key, when supplied in later study stages, is sent as a query
    parameter but is excluded from persisted query metadata and URLs.
    """
    if not endpoint.startswith("https://comtradeapi.un.org/"):
        raise ValueError("Only official https://comtradeapi.un.org endpoints are allowed.")

    clean_params = {str(key): str(value) for key, value in params.items() if value is not None}
    request_params = dict(clean_params)
    if subscription_key:
        request_params["subscription-key"] = subscription_key

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with requests.Session() as session:
                response = session.get(
                    endpoint,
                    params=request_params,
                    timeout=90,
                    headers={"Accept": "application/json"},
                )
                if response.status_code in RETRY_STATUS_CODES:
                    raise RuntimeError(
                        f"UN Comtrade returned retryable HTTP {response.status_code}."
                    )
                response.raise_for_status()
                payload = response.json()
                digest = canonical_json_sha256(payload)
                return ComtradeJSONResponse(
                    endpoint=redact_subscription_key(str(response.url)).split("?", maxsplit=1)[0],
                    query=clean_params,
                    retrieved_at_utc=datetime.now(UTC).isoformat(),
                    canonical_sha256=digest,
                    payload=payload,
                )
        except Exception as exc:  # bounded retry around an external official host
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                break
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"Could not retrieve official UN Comtrade JSON after {MAX_ATTEMPTS} attempts."
    ) from last_error
