#!/usr/bin/env python
"""Quick fixes for notebook execution issues identified in testing."""

from pathlib import Path

import numpy as np
import pandas as pd

AB_TESTING_DIR = Path("projects/ab_testing")
AB_TESTING_DATA_DIR = AB_TESTING_DIR / "data"


def create_sample_data_files():
    """Create sample data files for notebooks that need them."""
    # Create directories if they don't exist
    AB_TESTING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate sample A/B testing data
    print("Creating sample A/B testing datasets...")

    # Landing page data
    np.random.seed(42)
    landing_page_data = pd.DataFrame(
        {
            "user_id": range(1, 10001),
            "timestamp": pd.date_range("2024-01-01", periods=10000, freq="1min"),
            "group": np.random.choice(["control", "treatment"], 10000),
            "landing_page": np.random.choice(["old_page", "new_page"], 10000),
            "converted": np.random.choice([0, 1], 10000, p=[0.88, 0.12]),
        }
    )
    landing_page_data.to_csv(AB_TESTING_DATA_DIR / "landing_page.csv", index=False)
    print("  [OK] Created landing_page.csv")

    # Cookie Cats data (game A/B test)
    cookie_cats_data = pd.DataFrame(
        {
            "userid": range(1, 5001),
            "version": np.random.choice(["gate_30", "gate_40"], 5000),
            "sum_gamerounds": np.random.poisson(50, 5000),
            "retention_1": np.random.choice([True, False], 5000, p=[0.45, 0.55]),
            "retention_7": np.random.choice([True, False], 5000, p=[0.19, 0.81]),
        }
    )
    cookie_cats_data.to_csv(AB_TESTING_DATA_DIR / "cookie_cats.csv", index=False)
    print("  [OK] Created cookie_cats.csv")

    # Pricing A/B test data
    pricing_data = pd.DataFrame(
        {
            "user_id": range(1, 2001),
            "group": np.random.choice(["control", "treatment"], 2000),
            "price_shown": np.random.choice([9.99, 14.99], 2000),
            "purchased": np.random.choice([0, 1], 2000, p=[0.85, 0.15]),
            "revenue": np.where(
                np.random.choice([0, 1], 2000, p=[0.85, 0.15]) == 1,
                np.random.choice([9.99, 14.99], 2000),
                0,
            ),
        }
    )
    pricing_data.to_csv(AB_TESTING_DATA_DIR / "pricing_ab.csv", index=False)
    print("  [OK] Created pricing_ab.csv")


def fix_syntax_errors():
    """Provide guidance for fixing syntax errors."""
    print("\n" + "=" * 60)
    print("MANUAL FIXES REQUIRED")
    print("=" * 60)
    print("\nSyntax errors found in:")
    print("1. projects/ab_testing/ab_testing_ecommerce_playbook.ipynb - Line 7")
    print("   -> Check for unterminated strings, missing quotes")
    print("\n2. experiment_story_templates.ipynb - Line 52")
    print("   -> Check for unterminated strings, f-string issues")
    print("\nTo fix:")
    print("1. Open each notebook in Jupyter")
    print("2. Navigate to the indicated line")
    print("3. Look for missing closing quotes or brackets")
    print("4. Save the notebook after fixing")


def fix_import_issues():
    """Create a script to fix import issues."""
    print("\n" + "=" * 60)
    print("IMPORT FIXES")
    print("=" * 60)

    fix_script = """
# Add this to notebooks using erfcinv:
try:
    from scipy.special import erfcinv
except ImportError:
    # Fallback to erfinv if erfcinv not available
    from scipy.special import erfinv
    import numpy as np
    def erfcinv(x):
        return -np.sqrt(2) * erfinv(1 - x)
"""

    print("For math.erfcinv errors, add to the notebook:")
    print(fix_script)

    # Save the fix script
    with open(AB_TESTING_DIR / "erfcinv_fix.py", "w") as f:
        f.write(fix_script)
    print("\n[OK] Saved fix script to projects/ab_testing/erfcinv_fix.py")


def main():
    """Run all fixes."""
    print("=" * 60)
    print("NOTEBOOK ISSUE FIXER")
    print("=" * 60)

    # Check current directory
    if not AB_TESTING_DIR.exists():
        print("ERROR: Run this script from the project root directory!")
        return

    # Create sample data files
    create_sample_data_files()

    # Provide syntax error guidance
    fix_syntax_errors()

    # Create import fix script
    fix_import_issues()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Review and fix the syntax errors manually")
    print("2. Add the erfcinv import fix to affected notebooks")
    print("3. Re-run the tests:")
    print("   python scripts/run_notebook_tests.py --dirs projects/ab_testing")
    print("\n[OK] Quick fixes complete!")


if __name__ == "__main__":
    main()
