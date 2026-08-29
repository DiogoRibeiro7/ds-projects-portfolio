"""Probe the public INE mirror payload without changing the empirical pipeline."""

from __future__ import annotations

import json
from typing import Any

import requests

ENDPOINT = "https://gateway.pipeworx.io/ine-pt/mcp"


def _call(name: str, arguments: dict[str, object]) -> Any:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    response = requests.post(ENDPOINT, json=payload, timeout=30.0)
    response.raise_for_status()
    body = response.json()
    text = body["result"]["content"][0]["text"]
    return json.loads(text)


def _lisbon_municipality_code(metadata: list[dict[str, Any]]) -> str:
    categories = metadata[0]["Dimensoes"]["Categoria_Dim"]
    matches: list[str] = []
    for group in categories:
        for entries in group.values():
            for entry in entries:
                if (
                    str(entry.get("dim_num")) == "2"
                    and str(entry.get("categ_dsg", "")).strip().casefold() == "lisboa"
                    and str(entry.get("categ_nivel")) == "5"
                ):
                    matches.append(str(entry["categ_cod"]))
    if len(matches) != 1:
        raise ValueError(f"Expected one Lisbon municipality code, found {matches!r}")
    return matches[0]


def main() -> None:
    """Print metadata and municipality-filtered data shapes for one INE indicator."""
    metadata = _call("indicator_meta", {"varcd": "0009631", "lang": "PT"})
    lisboa_code = _lisbon_municipality_code(metadata)
    print(f"Lisboa municipality Dim2 code: {lisboa_code}")
    values = _call(
        "get_indicator",
        {"varcd": "0009631", "dims": {"Dim2": lisboa_code}, "lang": "PT"},
    )
    print(json.dumps(values, ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
