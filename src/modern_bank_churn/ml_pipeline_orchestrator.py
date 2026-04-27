from __future__ import annotations

import sys
from importlib import util
from pathlib import Path

_base = (
    Path(__file__).resolve().parents[2]
    / "archive"
    / "legacy"
    / "projects"
    / "mlops"
    / "modern_bank_churn"
    / "legacy"
    / "modern-bank-churn"
)
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))

_source = _base / "ml_pipeline_orchestrator.py"
_spec = util.spec_from_file_location(
    "modern_bank_churn._ml_pipeline_orchestrator", _source
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load module from {_source}")
_module = util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

sys.modules[__name__] = _module
