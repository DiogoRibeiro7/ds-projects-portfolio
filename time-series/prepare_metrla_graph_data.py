#!/usr/bin/env python
"""
Prepare METR-LA traffic data for spatio-temporal GNN models.

This script:
- Loads traffic time series from `metr-la.h5`.
- Loads adjacency matrix from `adj_mx.pkl`.
- Writes a single `graph_traffic.npz` file containing:
    - traffic: (T, N) float32 array
    - adjacency: (N, N) float32 array

Usage:
    python prepare_metrla_graph_data.py \
        --input-dir path/to/raw/files \
        --output-path data/graph_traffic.npz \
        [--h5-dataset-name speed]

Assumptions:
- `metr-la.h5` and `adj_mx.pkl` live in the same `--input-dir`.
- `adj_mx.pkl` is in the standard DCRNN format:
    (sensor_ids, sensor_id_to_ind, adj_mx)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pickle


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for data preparation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - input_dir: directory where METR-LA files live.
        - output_path: where to write graph_traffic.npz.
        - h5_dataset_name: optional name of dataset inside metr-la.h5.
    """
    parser = argparse.ArgumentParser(
        description="Prepare METR-LA data into graph_traffic.npz for ST-GCN notebook."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing metr-la.h5 and adj_mx.pkl.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/graph_traffic.npz",
        help="Output .npz file path (default: data/graph_traffic.npz).",
    )
    parser.add_argument(
        "--h5-dataset-name",
        type=str,
        default=None,
        help="Optional explicit name of dataset inside metr-la.h5 (e.g. 'speed').",
    )

    return parser.parse_args()


def discover_h5_dataset_name(h5_file: h5py.File) -> str:
    """Try to automatically pick the correct dataset in a METR-LA .h5 file.

    Heuristics:
    - Prefer a dataset whose name contains 'speed' (case-insensitive).
    - Otherwise, pick the first dataset with 2 or 3 dimensions.

    Parameters
    ----------
    h5_file : h5py.File
        Open HDF5 file handle.

    Returns
    -------
    str
        Chosen dataset name.

    Raises
    ------
    RuntimeError
        If no suitable dataset is found.
    """
    candidates: List[str] = list(h5_file.keys())
    if not candidates:
        raise RuntimeError("No datasets found inside metr-la.h5.")

    # Prefer datasets that contain 'speed' in the name
    for name in candidates:
        if "speed" in name.lower():
            return name

    # Otherwise, choose first dataset with 2D or 3D shape
    for name in candidates:
        ds = h5_file[name]
        if isinstance(ds, h5py.Dataset) and ds.ndim in (2, 3):
            return name

    raise RuntimeError(
        f"Could not infer dataset name from metr-la.h5. Available keys: {candidates}"
    )


def load_traffic_from_h5(
    h5_path: Path, dataset_name: Optional[str] = None
) -> np.ndarray:
    """Load METR-LA traffic tensor from an HDF5 file.

    Parameters
    ----------
    h5_path : Path
        Path to metr-la.h5.
    dataset_name : Optional[str]
        If provided, use this dataset name explicitly.
        If None, try to discover it automatically.

    Returns
    -------
    np.ndarray
        Traffic array of shape (T, N) or (T, N, 1) as float32.

    Raises
    ------
    FileNotFoundError
        If the HDF5 file does not exist.
    RuntimeError
        If a suitable dataset cannot be found or has invalid shape.
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"metr-la.h5 not found at: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if dataset_name is None:
            dataset_name = discover_h5_dataset_name(f)

        if dataset_name not in f:
            raise RuntimeError(
                f"Dataset '{dataset_name}' not found in {h5_path}. "
                f"Available keys: {list(f.keys())}"
            )

        data = f[dataset_name][:]
        if data.ndim not in (2, 3):
            raise RuntimeError(
                f"Expected 2D or 3D data, got shape {data.shape} in dataset '{dataset_name}'."
            )

    traffic = np.array(data, dtype=np.float32)
    return traffic


def load_adjacency_from_pkl(pkl_path: Path) -> np.ndarray:
    """Load adjacency matrix from a DCRNN-style adj_mx.pkl file.

    The file is expected to contain a tuple/list:
        (sensor_ids, sensor_id_to_ind, adj_mx)

    Parameters
    ----------
    pkl_path : Path
        Path to adj_mx.pkl.

    Returns
    -------
    np.ndarray
        Adjacency matrix of shape (N, N) as float32.

    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist.
    RuntimeError
        If the loaded structure does not match expectations.
    """
    if not pkl_path.exists():
        raise FileNotFoundError(f"adj_mx.pkl not found at: {pkl_path}")

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    # DCRNN-style: (sensor_ids, sensor_id_to_ind, adj_mx)
    if isinstance(obj, (list, tuple)) and len(obj) == 3:
        _, _, adj_mx = obj
    else:
        raise RuntimeError(
            "adj_mx.pkl does not have expected structure "
            "(sensor_ids, sensor_id_to_ind, adj_mx)."
        )

    adjacency = np.array(adj_mx, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise RuntimeError(
            f"Adjacency matrix must be square; got shape {adjacency.shape}."
        )

    return adjacency


def ensure_traffic_2d(traffic: np.ndarray) -> np.ndarray:
    """Ensure traffic has shape (T, N), dropping a trailing singleton dim if present.

    Parameters
    ----------
    traffic : np.ndarray
        Raw traffic array, typically (T, N) or (T, N, 1).

    Returns
    -------
    np.ndarray
        Traffic array of shape (T, N).
    """
    if traffic.ndim == 3 and traffic.shape[-1] == 1:
        traffic = traffic[..., 0]

    if traffic.ndim != 2:
        raise ValueError(
            f"Expected traffic array with 2 dimensions (T, N), got shape {traffic.shape}."
        )

    return traffic


def save_graph_npz(
    traffic: np.ndarray, adjacency: np.ndarray, output_path: Path
) -> None:
    """Save traffic and adjacency to a single .npz file.

    Parameters
    ----------
    traffic : np.ndarray
        Traffic array of shape (T, N), float32.
    adjacency : np.ndarray
        Adjacency matrix of shape (N, N), float32.
    output_path : Path
        Destination .npz path.
    """
    if traffic.shape[1] != adjacency.shape[0]:
        raise ValueError(
            f"Mismatch between traffic N={traffic.shape[1]} and adjacency N={adjacency.shape[0]}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        traffic=traffic.astype(np.float32),
        adjacency=adjacency.astype(np.float32),
    )


def main() -> None:
    """Entry point for the METR-LA graph data preparation script."""
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    h5_dataset_name: Optional[str] = args.h5_dataset_name

    h5_path = input_dir / "metr-la.h5"
    adj_pkl_path = input_dir / "adj_mx.pkl"

    print(f"Loading traffic from: {h5_path}")
    traffic = load_traffic_from_h5(h5_path, dataset_name=h5_dataset_name)
    traffic = ensure_traffic_2d(traffic)
    print(f"Traffic array shape (T, N): {traffic.shape}")

    print(f"Loading adjacency from: {adj_pkl_path}")
    adjacency = load_adjacency_from_pkl(adj_pkl_path)
    print(f"Adjacency matrix shape (N, N): {adjacency.shape}")

    print(f"Saving combined graph data to: {output_path}")
    save_graph_npz(traffic, adjacency, output_path)

    print("Done. You can now point the GNN notebook to this file:")
    print(f"  {output_path.resolve()}")


if __name__ == "__main__":
    main()
