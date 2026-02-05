#!/usr/bin/env python
"""
Comprehensive documentation build script for the Data Science Portfolio.

This script automates the entire documentation generation process including:
- API reference generation
- Notebook conversion
- Example gallery creation
- HTML and PDF output
"""

import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DocumentationBuilder:
    """Comprehensive documentation builder for the project."""

    def __init__(self, docs_dir: Path = None, clean: bool = True):
        """Initialize the documentation builder.

        Parameters
        ----------
        docs_dir : Path, optional
            Documentation directory path.
        clean : bool, default=True
            Whether to clean previous builds.
        """
        self.docs_dir = docs_dir or Path(__file__).parent
        self.build_dir = self.docs_dir / "_build"
        self.api_dir = self.docs_dir / "api"
        self.clean = clean

        # Ensure directories exist
        self.api_dir.mkdir(exist_ok=True)

    def clean_build(self):
        """Clean previous build artifacts."""
        if self.build_dir.exists() and self.clean:
            print("🧹 Cleaning previous build...")
            shutil.rmtree(self.build_dir)

    def generate_api_reference(self):
        """Generate API reference documentation from docstrings."""
        print("📚 Generating API reference...")

        # Define modules to document
        modules = {
            "statistics": {
                "title": "Statistical Methods",
                "modules": [
                    "src.statistics.core",
                    "src.statistics.bayesian",
                    "src.statistics.power_analysis",
                ],
            },
            "data_processing": {
                "title": "Data Processing",
                "modules": [
                    "src.data_processing.cleaning",
                    "src.data_processing.benchmark_performance",
                    "src.data_processing.optimization_examples",
                ],
            },
            "utils": {
                "title": "Utilities",
                "modules": [
                    "src.utils.caching",
                    "src.utils.cache_integration",
                    "src.utils.constants",
                    "src.utils.observability",
                ],
            },
            "api": {
                "title": "Machine Learning API",
                "modules": [
                    "src.api.ml_api",
                ],
            },
            "cloud": {
                "title": "Cloud Integrations",
                "modules": [
                    "src.cloud.cloud_integrations",
                ],
            },
            "ab_testing": {
                "title": "A/B Testing",
                "modules": ["ab_testing", "ab_testing.erfcinv_fix"],
            },
            "dashboard": {
                "title": "Dashboard Framework",
                "modules": [
                    "dashboard_enhanced.app",
                    "dashboard_enhanced.dashboard_framework",
                    "dashboard_enhanced.graphql_api",
                ],
            },
        }

        # Create main API index
        index_content = """API Reference
=============

This section contains the complete API reference for all modules in the Data Science Portfolio.

.. toctree::
   :maxdepth: 2
   :caption: API Modules

"""

        for category, info in modules.items():
            index_content += f"   {category}\n"

            # Create category file
            category_file = self.api_dir / f"{category}.rst"
            category_content = f"""{info["title"]}
{"=" * len(info["title"])}

.. currentmodule:: {category}

.. autosummary::
   :toctree: generated
   :recursive:

"""

            for module in info["modules"]:
                category_content += f"   {module}\n"

            category_file.write_text(category_content)
            print(f"  ✓ Generated {category}.rst")

        # Write API index
        api_index = self.api_dir / "index.rst"
        api_index.write_text(index_content)
        print("  ✓ Generated API index")

    def generate_module_docs(self):
        """Generate individual module documentation files."""
        print("📝 Generating module documentation...")

        # Template for module documentation
        template = """.. _{module_name}:

{module_title}
{title_underline}

.. automodule:: {module_path}
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

.. rubric:: Classes

.. autosummary::
   :toctree: generated
   :template: class.rst

   {classes}

.. rubric:: Functions

.. autosummary::
   :toctree: generated
   :template: function.rst

   {functions}
"""

        # This would need to be expanded with actual module introspection
        # For now, we'll create placeholders
        modules_to_document = [
            "src.statistics.core",
            "src.data_processing.cleaning",
            "src.utils.caching",
        ]

        for module_path in modules_to_document:
            module_name = module_path.split(".")[-1]
            doc_file = self.api_dir / f"{module_name}_detail.rst"

            content = template.format(
                module_name=module_name,
                module_title=module_name.replace("_", " ").title(),
                title_underline="=" * len(module_name),
                module_path=module_path,
                classes="",  # Would be populated with actual classes
                functions="",  # Would be populated with actual functions
            )

            doc_file.write_text(content)
            print(f"  ✓ Generated {module_name}_detail.rst")

    def convert_notebooks(self):
        """Convert Jupyter notebooks to documentation."""
        print("📓 Converting notebooks...")

        notebooks_dirs = [
            PROJECT_ROOT / "ab_testing",
            PROJECT_ROOT / "modern-bank-churn",
            PROJECT_ROOT / "time-series",
        ]

        output_dir = self.docs_dir / "notebooks"
        output_dir.mkdir(exist_ok=True)

        for nb_dir in notebooks_dirs:
            if not nb_dir.exists():
                continue

            for notebook in nb_dir.glob("*.ipynb"):
                print(f"  Converting {notebook.name}...")
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "jupyter",
                            "nbconvert",
                            "--to",
                            "rst",
                            "--output-dir",
                            str(output_dir),
                            str(notebook),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    print(f"    ✓ Converted {notebook.name}")
                except subprocess.CalledProcessError as e:
                    print(f"    ✗ Failed to convert {notebook.name}: {e}")

    def create_example_gallery(self):
        """Create an example gallery from Python scripts."""
        print("🎨 Creating example gallery...")

        examples_dir = PROJECT_ROOT / "examples"
        gallery_dir = self.docs_dir / "gallery"
        gallery_dir.mkdir(exist_ok=True)

        if not examples_dir.exists():
            examples_dir.mkdir(exist_ok=True)
            # Create a sample example
            example_file = examples_dir / "example_ab_testing.py"
            example_file.write_text('''"""
A/B Testing Example
==================

This example demonstrates how to perform A/B testing analysis.
"""

import pandas as pd
import numpy as np
from src.statistics.core import two_prop_ztest

# Generate sample data
np.random.seed(42)
n_control = 1000
n_treatment = 1000

# Simulate conversion data
control_conversions = np.random.binomial(1, 0.10, n_control).sum()
treatment_conversions = np.random.binomial(1, 0.12, n_treatment).sum()

# Perform statistical test
z_stat, p_value = two_prop_ztest(
    control_conversions, n_control,
    treatment_conversions, n_treatment
)

print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at α=0.05: {p_value < 0.05}")
''')
            print("  ✓ Created sample example")

    def build_html(self):
        """Build HTML documentation."""
        print("🏗️ Building HTML documentation...")

        cmd = [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            str(self.docs_dir),
            str(self.build_dir / "html"),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print("  ✓ HTML documentation built successfully")
            print(f"  📁 Output: {self.build_dir / 'html' / 'index.html'}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ HTML build failed: {e}")
            print(f"    Error output: {e.stderr.decode()}")

    def build_pdf(self):
        """Build PDF documentation (requires LaTeX)."""
        print("📄 Building PDF documentation...")

        cmd = [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "latex",
            str(self.docs_dir),
            str(self.build_dir / "latex"),
        ]

        try:
            # Build LaTeX
            subprocess.run(cmd, check=True, capture_output=True)

            # Convert to PDF
            latex_dir = self.build_dir / "latex"
            if sys.platform == "win32":
                pdf_cmd = ["pdflatex", "DataSciencePortfolio.tex"]
            else:
                pdf_cmd = ["make"]

            subprocess.run(pdf_cmd, cwd=latex_dir, check=True, capture_output=True)
            print("  ✓ PDF documentation built successfully")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  ⚠️ PDF build skipped (LaTeX not available): {e}")

    def check_links(self):
        """Check for broken links in documentation."""
        print("🔗 Checking documentation links...")

        cmd = [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "linkcheck",
            str(self.docs_dir),
            str(self.build_dir / "linkcheck"),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Parse results
            output_file = self.build_dir / "linkcheck" / "output.txt"
            if output_file.exists():
                with open(output_file) as f:
                    broken_links = [line for line in f if "broken" in line.lower()]

                if broken_links:
                    print(f"  ⚠️ Found {len(broken_links)} broken links")
                    for link in broken_links[:5]:  # Show first 5
                        print(f"    - {link.strip()}")
                else:
                    print("  ✓ All links valid")
            else:
                print("  ✓ Link check completed")

        except subprocess.CalledProcessError as e:
            print(f"  ✗ Link check failed: {e}")

    def create_requirements_file(self):
        """Create requirements file for documentation dependencies."""
        print("📦 Creating documentation requirements...")

        requirements = """# Documentation dependencies
sphinx>=4.5.0
sphinx-rtd-theme>=1.0.0
furo>=2023.0.0
pydata-sphinx-theme>=0.13.0
sphinx-copybutton>=0.5.0
sphinx-autodoc-typehints>=1.23.0
sphinxcontrib-mermaid>=0.8.0
myst-parser>=1.0.0
myst-nb>=0.17.0
nbsphinx>=0.9.0
jupyter>=1.0.0
"""

        req_file = self.docs_dir / "requirements-docs.txt"
        req_file.write_text(requirements)
        print(f"  ✓ Created {req_file.name}")

    def build_all(self):
        """Run the complete documentation build process."""
        print("\n" + "=" * 60)
        print("📚 COMPREHENSIVE DOCUMENTATION BUILD")
        print("=" * 60 + "\n")

        # Clean if requested
        if self.clean:
            self.clean_build()

        # Generate all documentation
        self.generate_api_reference()
        self.generate_module_docs()
        self.convert_notebooks()
        self.create_example_gallery()
        self.create_requirements_file()

        # Build outputs
        self.build_html()
        self.build_pdf()

        # Check quality
        self.check_links()

        print("\n" + "=" * 60)
        print("✅ Documentation build complete!")
        print("=" * 60)

        # Open documentation
        index_path = self.build_dir / "html" / "index.html"
        if index_path.exists():
            print(f"\n📖 View documentation at: file://{index_path}")

            if sys.platform == "win32":
                os.startfile(index_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(index_path)])
            else:
                subprocess.run(["xdg-open", str(index_path)])


def main():
    """Main entry point for documentation builder."""
    parser = argparse.ArgumentParser(
        description="Build comprehensive documentation for Data Science Portfolio"
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Skip cleaning previous builds"
    )
    parser.add_argument(
        "--html-only", action="store_true", help="Build only HTML documentation"
    )
    parser.add_argument(
        "--check-links", action="store_true", help="Only check for broken links"
    )

    args = parser.parse_args()

    # Create builder
    builder = DocumentationBuilder(clean=not args.no_clean)

    if args.check_links:
        builder.check_links()
    elif args.html_only:
        builder.generate_api_reference()
        builder.build_html()
    else:
        builder.build_all()


if __name__ == "__main__":
    main()
