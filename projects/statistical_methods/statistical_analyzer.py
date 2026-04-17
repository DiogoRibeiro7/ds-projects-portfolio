"""Statistical analyzer used by integration tests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StatisticalAnalyzer:
    """Lightweight statistical summary helper for integration tests."""

    data: pd.DataFrame

    def generate_summary(self) -> pd.DataFrame:
        """Generate a descriptive statistics summary."""
        if self.data is None:
            raise ValueError("data is required")
        return self.data.describe(include="all")
