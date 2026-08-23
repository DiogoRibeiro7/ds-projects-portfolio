"""Sphinx configuration for the documentation site.

This file exposes the knobs Sphinx reads when building HTML/LaTeX output and
configures every extension we rely on for notebooks, diagrams, and API docs.
"""

import asyncio
import os
import sys
import types

REPO_ROOT = os.path.abspath("..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# Legacy project content is archived under archive/legacy/projects.

# ---------------------------------------------------------------------------
# Lightweight stubs for optional third-party frameworks used in the API
# ---------------------------------------------------------------------------


def _install_stub(name: str, module: types.ModuleType) -> None:
    """Register a stub module if the real dependency is unavailable."""
    if name not in sys.modules:
        sys.modules[name] = module


# uvloop stub (Windows runners do not provide uvloop)
class _UvloopPolicy(asyncio.DefaultEventLoopPolicy):
    """Dummy uvloop-compatible policy."""


uvloop_stub = types.ModuleType("uvloop")
uvloop_stub.EventLoopPolicy = _UvloopPolicy
_install_stub("uvloop", uvloop_stub)


# Graphene stub
graphene_stub = types.ModuleType("graphene")


class _GrapheneBase:
    def __init__(self, *args, **kwargs):
        pass


class _GrapheneMutation(_GrapheneBase):
    @classmethod
    def Field(cls, *args, **kwargs):
        return None


class _GrapheneObjectType(_GrapheneBase):
    pass


def _graphene_field(*args, **kwargs):
    return None


class _GrapheneSchema:
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, *args, **kwargs):
        return None


graphene_stub.Mutation = _GrapheneMutation
graphene_stub.ObjectType = _GrapheneObjectType
graphene_stub.InputObjectType = _GrapheneBase
graphene_stub.Field = _graphene_field
graphene_stub.List = lambda *args, **kwargs: None
graphene_stub.Boolean = graphene_stub.DateTime = graphene_stub.Float = (
    graphene_stub.Int
) = graphene_stub.JSONString = graphene_stub.String = _GrapheneBase
graphene_stub.Schema = _GrapheneSchema
_install_stub("graphene", graphene_stub)


# Flask + related extensions stubs
flask_stub = types.ModuleType("flask")


class _Flask:
    def __init__(self, *args, **kwargs):
        self.config = {}

    def route(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def test_client(self):
        class _Client:
            def get(self, *args, **kwargs):
                return None

        return _Client()


def _identity(*args, **kwargs):
    return None


flask_stub.Flask = _Flask
flask_stub.jsonify = _identity
flask_stub.render_template = _identity
flask_stub.request = types.SimpleNamespace()
flask_stub.send_file = _identity
flask_stub.session = {}
_install_stub("flask", flask_stub)

flask_cors_stub = types.ModuleType("flask_cors")
flask_cors_stub.CORS = lambda *args, **kwargs: None
_install_stub("flask_cors", flask_cors_stub)

flask_jwt_stub = types.ModuleType("flask_jwt_extended")


class _JWTManager:
    def __init__(self, *args, **kwargs):
        pass


def _jwt_required(fn=None, *args, **kwargs):
    def decorator(func):
        return func

    return decorator if fn is None else fn


flask_jwt_stub.JWTManager = _JWTManager
flask_jwt_stub.create_access_token = lambda *args, **kwargs: "token"
flask_jwt_stub.get_jwt_identity = lambda: "user"
flask_jwt_stub.jwt_required = _jwt_required
_install_stub("flask_jwt_extended", flask_jwt_stub)

flask_limiter_stub = types.ModuleType("flask_limiter")


class _Limiter:
    def __init__(self, *args, **kwargs):
        pass


flask_limiter_stub.Limiter = _Limiter
_install_stub("flask_limiter", flask_limiter_stub)

flask_limiter_util = types.ModuleType("flask_limiter.util")
flask_limiter_util.get_remote_address = lambda *args, **kwargs: "127.0.0.1"
sys.modules["flask_limiter.util"] = flask_limiter_util

flask_socketio_stub = types.ModuleType("flask_socketio")


class _SocketIO:
    def __init__(self, *args, **kwargs):
        pass

    def emit(self, *args, **kwargs):
        pass


flask_socketio_stub.SocketIO = _SocketIO
flask_socketio_stub.emit = lambda *args, **kwargs: None
_install_stub("flask_socketio", flask_socketio_stub)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Data Science Projects Portfolio"
copyright = "2024, Diogo Ribeiro"
author = "Diogo Ribeiro"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.autosummary",
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
    "api_reference.md",
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

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Autosummary settings
autosummary_generate = True

autodoc_mock_imports = [
    "fastapi",
    "uvicorn",
    "starlette",
    "aioredis",
    "redis",
    "redis.asyncio",
    "fakeredis",
    "graphql",
    "prometheus_client",
    "mlflow",
    "shap",
    "boto3",
    "botocore",
    "sagemaker",
    "google",
    "google.cloud",
    "google.oauth2",
    "azure",
    "azure.storage.blob",
    "azure.ai.ml",
    "azure.identity",
    "azureml",
    "azureml.core",
    "opentelemetry",
    "opentelemetry.exporter",
    "opentelemetry.sdk",
    "opentelemetry.instrumentation",
    "opentelemetry.sdk.trace",
    "opentelemetry.sdk.metrics",
    "opentelemetry.sdk.resources",
    "xgboost",
]

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
