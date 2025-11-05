"""Setup configuration for data science portfolio package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README = Path(__file__).parent / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

# Read requirements
REQUIREMENTS = Path(__file__).parent / "requirements.txt"
install_requires = []
if REQUIREMENTS.exists():
    install_requires = [
        line.strip() 
        for line in REQUIREMENTS.read_text().splitlines() 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ds-portfolio",
    version="1.0.0",
    author="Diogo Ribeiro",
    author_email="dfr@esmad.ipp.pt",
    description="Professional data science portfolio with statistical analysis and experimentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diogoribeiro7/data-science-portfolio",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "altair>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ds-portfolio=ds_portfolio.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "data science",
        "statistics",
        "a/b testing",
        "experimentation", 
        "machine learning",
        "analytics",
        "portfolio",
    ],
    project_urls={
        "Bug Reports": "https://github.com/diogoribeiro7/data-science-portfolio/issues",
        "Source": "https://github.com/diogoribeiro7/data-science-portfolio",
        "Documentation": "https://diogoribeiro7.github.io/data-science-portfolio/",
    },
)
