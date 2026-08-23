"""Sphinx configuration for the portfolio documentation site."""


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Data Science Projects Portfolio"
copyright = "2024, Diogo Ribeiro"
author = "Diogo Ribeiro"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.ifconfig",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "modules/*",
]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom CSS
html_css_files = [
    "custom.css",
]

# -- Extension configuration -------------------------------------------------

# MyST settings for Markdown support
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# nbsphinx settings for Jupyter notebooks
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_timeout = 600

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
}

# Copy button settings
copybutton_prompt_text = r">>> |\\$ |In \\[\\d*\\]: | {2,5}\\.\\.\\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Mermaid settings for diagrams
mermaid_output_format = "raw"
mermaid_version = "latest"

# TODO extension settings
todo_include_todos = True

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": r"""
        \\usepackage{amsmath}
        \\usepackage{amssymb}
        \\usepackage{amsfonts}
    """,
}

# Grouping the document tree into LaTeX files
latex_documents = [
    (
        master_doc,
        "DataSciencePortfolio.tex",
        "Data Science Portfolio Documentation",
        author,
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (
        master_doc,
        "datascienceportfolio",
        "Data Science Portfolio Documentation",
        [author],
        1,
    )
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "DataSciencePortfolio",
        "Data Science Portfolio Documentation",
        author,
        "DataSciencePortfolio",
        "Comprehensive data science portfolio with ML pipelines, statistical methods, and dashboards.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------

epub_title = project
epub_exclude_files = ["search.html"]

# -- Custom setup ------------------------------------------------------------


def setup(app):
    """Register custom build steps during Sphinx initialization."""
    app.add_css_file("custom.css")
