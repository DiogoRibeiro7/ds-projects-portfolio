"""Probe the public INE mirror payload without changing the empirical pipeline."""

from __future__ import annotations

import json

import requests

ENDPOINT = "https://gateway.pipeworx.io/ine-pt/mcp"


def main() -> None:
    """Print the response shape for one known INE indicator metadata request."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "indicator_meta",
            "arguments": {"varcd": "0009631", "lang": "PT"},
        },
    }
    response = requests.post(ENDPOINT, json=payload, timeout=30.0)
    response.raise_for_status()
    body = response.json()
    print(json.dumps(body, ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
