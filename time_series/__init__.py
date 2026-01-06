"""
Compatibility package for time-series utilities.

The repository stores time-series scripts in ``time-series/``.  This namespace
module exposes them under ``time_series`` so that imports and autodoc tooling
work consistently across platforms.
"""

from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parent
_LEGACY_DIR = _PACKAGE_ROOT.parent / "time-series"

if _LEGACY_DIR.exists():
    __path__.append(str(_LEGACY_DIR))
    if str(_LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(_LEGACY_DIR))
