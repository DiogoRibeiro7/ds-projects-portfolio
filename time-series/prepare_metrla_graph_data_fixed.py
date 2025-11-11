#!/usr/bin/env python
"""
Prepare METR-LA traffic data for the GNN notebook.

Assumes:
    - Raw files in: /data/METR-LA/
        * metr-la.h5
        * adj_mx.pkl
    - Writes: data/graph_traffic.npz  (relative to project root)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pickle


RAW_DIR = Path("/data/METR-LA")
OUTPUT_PATH = Path("data") / "graph_traffic.npz"


def discover_h5_dataset_name(h5_file: h5py.File) -> str:
    """Heuristic to pick the dataset inside metr-la.h5."""
    keys: List[str] = list(h5_file.keys())
    if not keys:
        raise RuntimeError("No datasets found in metr-la.h5")

    # Prefer something with 'speed' in the name
    for name in keys:
        if "speed" in name.lower():
            return name

    # Otherwise first 2D/3D dataset
    for name in keys:
        ds = h5_file[name]
        if isinstance(ds, h5py.Dataset) and ds.ndim in (2, 3):
            return name

    raise RuntimeError(f"Could not infer dataset; keys: {keys}")


def load_traffic(h5_path: Path, dataset_name: Optional[str] = None) -> np.ndarray:
    """Load traffic tensor from metr-la.h5."""
    if not h5_path.exists():
        raise FileNotFoundError(f"metr-la.h5 not found at {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if dataset_name is None:
            dataset_name = discover_h5_dataset_name(f)

        if dataset_name not in f:
            raise RuntimeError(
                f"Dataset '{dataset_name}' not found. Keys: {list(f.keys())}"
            )

        data = f[dataset_name][:]
        if data.ndim not in (2, 3):
            raise RuntimeError(f"Unexpected shape for dataset '{dataset_name}': {data.shape}")

    traffic = np.array(data, dtype=np.float32)
    if traffic.ndim == 3 and traffic.shape[-1] == 1:
        traffic = traffic[..., 0]
    if traffic.ndim != 2:
        raise ValueError(f"Expected (T, N); got {traffic.shape}")
    return traffic


def load_adjacency(pkl_path: Path) -> np.ndarray:
    """Load adjacency matrix from adj_mx.pkl."""
    if not pkl_path.exists():
        raise FileNotFoundError(f"adj_mx.pkl not found at {pkl_path}")

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    if isinstance(obj, (list, tuple)) and len(obj) == 3:
        _, _, adj_mx = obj
    else:
        raise RuntimeError("Unexpected structure in adj_mx.pkl; expected (ids, map, adj_mx).")

    adj = np.array(adj_mx, dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square; got {adj.shape}")
    return adj


def main() -> None:
    h5_path = RAW_DIR / "metr-la.h5"
    adj_path = RAW_DIR / "adj_mx.pkl"

    print(f"Loading traffic from: {h5_path}")
    traffic = load_traffic(h5_path)  # or load_traffic(h5_path, dataset_name="speed")
    print("Traffic shape (T, N):", traffic.shape)

    print(f"Loading adjacency from: {adj_path}")
    adjacency = load_adjacency(adj_path)
    print("Adjacency shape (N, N):", adjacency.shape)

    if traffic.shape[1] != adjacency.shape[0]:
        raise ValueError(
            f"Mismatch between traffic N={traffic.shape[1]} and adjacency N={adjacency.shape[0]}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        traffic=traffic.astype(np.float32),
        adjacency=adjacency.astype(np.float32),
    )

    print("Saved graph data to:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
