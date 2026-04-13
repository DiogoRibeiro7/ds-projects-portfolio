"""Centralized logging utilities for pipeline and workflow modules."""

import logging
from typing import Any


def configure_pipeline_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt: str = "%Y-%m-%dT%H:%M:%S",
    force: bool = True,
) -> None:
    """Configure root logging for pipeline execution."""
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, force=force)


def get_pipeline_logger(name: str | None = None) -> logging.Logger:
    """Return a named logger for pipeline modules."""
    return logging.getLogger(name if name else __name__)
