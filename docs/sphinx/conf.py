"""Sphinx configuration for the cnbs-predictor documentation.

The API reference is generated with ``autodoc`` + ``napoleon`` from the
NumPy-style docstrings in ``src/``. Heavy/compiled dependencies (cfgrib,
netCDF4, boto3, plotly, ...) are mocked rather than installed, so the docs
build on a lightweight pip-only environment (locally and on Read the Docs)
without the compiled conda stack. Autodoc only needs to *import* the modules
to read their docstrings — it never runs the science.
"""

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# conf.py lives at docs/sphinx/; the importable package root is two levels up
# (the repo root, which contains the `src/` package).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

# -- Project information -----------------------------------------------------
project = "cnbs-predictor"
author = "Great Lakes AI Lab"
copyright = f"{datetime.now():%Y}, {author}"
release = "1.2.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy / Google docstring parsing
    "sphinx.ext.viewcode",      # "[source]" links next to each object
    "sphinx.ext.intersphinx",   # cross-link to numpy/pandas/python docs
    "myst_parser",              # allow Markdown sources alongside reST
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Both reST and Markdown sources are accepted.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,       # still list undocumented members (some are sparse)
    "show-inheritance": True,
    "member-order": "bysource",
}

# Compiled / heavy third-party deps imported at module scope in src/.
# Mocking these lets autodoc import the modules without the binaries present.
#
# IMPORTANT: do NOT mock anything that a *real* (installed) dependency needs.
# In particular pandas/numpy are installed for real (see docs/sphinx/
# requirements.txt) and pandas imports `dateutil` internally — mocking
# `dateutil` breaks pandas' C-extension init. So pandas, numpy, requests and
# dateutil are installed, not mocked; only the heavy/compiled libraries below
# are mocked.
autodoc_mock_imports = [
    "cfgrib",
    "netCDF4",
    "boto3",
    "botocore",
    "xarray",
    "scipy",
    "sklearn",
    "xgboost",
    "joblib",
    "matplotlib",
    "plotly",
    "cartopy",
    "properscoring",
    "bs4",
    "requests_toolbelt",
    "tabulate",
]

# -- Napoleon (docstring style) ----------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_rtype = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output -------------------------------------------------------------
# furo is a clean, modern theme; falls back gracefully if not installed.
try:
    import furo  # noqa: F401

    html_theme = "furo"
except ImportError:  # pragma: no cover
    html_theme = "alabaster"

html_title = "cnbs-predictor"
html_static_path = ["_static"]
