"""Execute the preregistered v2 runner after frozen-input verification."""

from __future__ import annotations

from pathlib import Path

import run_v2_stochastic as runner

from mobility_optimization.frozen_inputs import load_frozen_relocation_matrix

MATRIX_PATH = Path("evidence/v2_relocation_cost_matrix.csv")


def main() -> None:
    """Verify the frozen matrix, then execute the unchanged v2 study."""
    load_frozen_relocation_matrix(MATRIX_PATH)
    runner.main()


if __name__ == "__main__":
    main()
