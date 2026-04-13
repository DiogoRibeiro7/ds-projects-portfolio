#!/usr/bin/env python
"""Download sample datasets with checksum verification.

Usage:
    python scripts/download_sample_dataset.py --dataset iris
    python scripts/download_sample_dataset.py --dataset titanic
    python scripts/download_sample_dataset.py --all --output-dir data/samples
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SAMPLE_DATASETS: dict[str, dict[str, str]] = {
    "iris": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "filename": "iris.data",
        "checksum": "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0",
        "description": "Classic Iris dataset for classification experiments.",
    },
    "titanic": {
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "filename": "titanic.csv",
        "checksum": "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7",
        "description": "Titanic passenger dataset for survival modeling.",
    },
}

DEFAULT_OUTPUT_DIR = Path("data") / "samples"


def calculate_checksum(path: Path, algorithm: str = "sha256") -> str:
    hash_func = hashlib.new(algorithm)
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "python-urllib/3"})
    try:
        with (
            urlopen(request, timeout=30) as response,
            destination.open("wb") as out_file,
        ):
            out_file.write(response.read())
    except HTTPError as exc:
        raise RuntimeError(
            f"HTTP error while downloading {url}: {exc.code} {exc.reason}"
        )
    except URLError as exc:
        raise RuntimeError(f"URL error while downloading {url}: {exc.reason}")


def verify_checksum(path: Path, expected_checksum: str) -> bool:
    actual = calculate_checksum(path)
    return actual.lower() == expected_checksum.lower()


def download_dataset(name: str, output_dir: Path, force: bool = False) -> Path:
    if name not in SAMPLE_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    metadata = SAMPLE_DATASETS[name]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / metadata["filename"]

    if destination.exists() and not force:
        print(f"Found existing file: {destination}")
        if verify_checksum(destination, metadata["checksum"]):
            print("Checksum verified. Skipping download.")
            return destination
        print("Checksum mismatch detected. Re-downloading.")

    print(f"Downloading {name}: {metadata['description']}")
    print(f"From: {metadata['url']}")
    download_file(metadata["url"], destination)

    if not verify_checksum(destination, metadata["checksum"]):
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file checksum does not match expected value for {name}."
        )

    print(f"Saved {destination} (sha256 verified)")
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download sample datasets and verify checksums."
    )
    parser.add_argument(
        "--dataset",
        choices=list(SAMPLE_DATASETS),
        help="The name of the sample dataset to download.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available sample datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save downloaded datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the file already exists and checksum matches.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.dataset and not args.all:
        print("Please specify --dataset or --all.")
        return 1

    selected = [args.dataset] if args.dataset else list(SAMPLE_DATASETS)

    for name in selected:
        try:
            download_dataset(name, args.output_dir, force=args.force)
        except Exception as exc:
            print(f"Failed to download {name}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
