"""
Compatibility package for the ``modern-bank-churn`` modules.

The original churn modeling utilities live under the ``modern-bank-churn/``
directory whose name cannot be imported directly.  This shim exposes the
modules under the canonical ``modern_bank_churn`` namespace and ensures the
legacy directory is discoverable on ``sys.path``.
"""

from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parent
_LEGACY_DIR = _PACKAGE_ROOT.parent / "modern-bank-churn"

# Allow ``import modern_bank_churn.<module>`` to locate the original sources.
if _LEGACY_DIR.exists():
    __path__.append(str(_LEGACY_DIR))
    if str(_LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(_LEGACY_DIR))
