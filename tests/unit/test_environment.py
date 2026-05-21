# tests/unit/test_environment.py
"""
Environment / dependency sanity checks.

These tests are guard rails against "works on my machine" drift:

1. Every package the project actually imports must be available.
2. Versions that are deliberately pinned in ``requirements/environment.yml``
   (cfgrib, scikit-learn, joblib, python) must match what's installed —
   this is parsed dynamically so the test stays in sync with the env file.

TensorFlow is intentionally not validated here: the env file pins
``tensorflow==2.20.0`` alongside ``tensorflow-intel==0.0.1`` (a
Windows-Intel-only stub), and the project is deployed on Windows but
developed on both Mac and Windows. Adding a TF check would create
platform-specific noise that doesn't reflect a real failure mode.
"""

import importlib
import importlib.metadata
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / "requirements" / "environment.yml"


# ---------------------------------------------------------------------------
# Environment-file parsing
# ---------------------------------------------------------------------------
def _parse_pins(env_path: Path) -> dict[str, str]:
    """
    Extract ``name==version`` pins from an environment.yml without needing
    PyYAML to parse the whole file (PyYAML *is* a project dep, but reading
    only the lines we care about avoids depending on YAML structure choices).
    """
    pins: dict[str, str] = {}
    pin_re = re.compile(r"^\s*-\s*([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-+]+)\s*$")
    for line in env_path.read_text().splitlines():
        m = pin_re.match(line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


PINS = _parse_pins(ENV_FILE)


# ---------------------------------------------------------------------------
# Importability — every package src/ relies on must be installed
# ---------------------------------------------------------------------------
# (distribution_name, import_name) — these differ for several packages.
REQUIRED_IMPORTS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("xarray", "xarray"),
    ("netcdf4", "netCDF4"),
    ("cfgrib", "cfgrib"),
    ("requests", "requests"),
    ("boto3", "boto3"),
    ("botocore", "botocore"),
    ("scikit-learn", "sklearn"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
    ("pyyaml", "yaml"),
    ("beautifulsoup4", "bs4"),
]


@pytest.mark.parametrize("dist_name,import_name", REQUIRED_IMPORTS)
def test_required_package_importable(dist_name, import_name):
    """Each required package can be imported (i.e. the env is complete)."""
    try:
        importlib.import_module(import_name)
    except ImportError as e:
        pytest.fail(
            f"Required package '{dist_name}' (import name '{import_name}') is not "
            f"installed or fails to import: {e}"
        )


# ---------------------------------------------------------------------------
# Version pins — versions deliberately fixed in environment.yml must match
# ---------------------------------------------------------------------------
# Maps the conda/pip package name (as it appears in environment.yml) to the
# distribution name we'll query via importlib.metadata. They're usually the
# same, but a few diverge (e.g. python isn't an installable distribution).
PINNED_PACKAGES = ["scikit-learn", "cfgrib", "joblib"]


@pytest.mark.parametrize("dist_name", PINNED_PACKAGES)
def test_pinned_version_matches_environment_yml(dist_name):
    """
    For every package pinned with '==' in environment.yml, the installed
    version must match. If the pin is bumped, update the env file (and
    rebuild) — this test catches unintentional drift.
    """
    if dist_name not in PINS:
        pytest.skip(f"{dist_name} is not pinned in environment.yml; nothing to check.")

    expected = PINS[dist_name]
    try:
        actual = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        pytest.fail(f"Pinned package '{dist_name}' is not installed.")

    assert actual == expected, (
        f"{dist_name} version mismatch: environment.yml pins '{expected}', "
        f"but installed version is '{actual}'."
    )


# ---------------------------------------------------------------------------
# Python version pin
# ---------------------------------------------------------------------------
def test_python_version_matches_pin():
    """Python interpreter version must match the pin in environment.yml."""
    if "python" not in PINS:
        pytest.skip("Python is not pinned in environment.yml.")

    expected = PINS["python"]  # e.g. "3.12.3"
    actual = ".".join(str(p) for p in sys.version_info[:3])
    assert actual == expected, (
        f"Python version mismatch: environment.yml pins '{expected}', "
        f"interpreter is '{actual}'."
    )


# ---------------------------------------------------------------------------
# Sanity check on the env-file parser itself
# ---------------------------------------------------------------------------
def test_environment_yml_exists_and_contains_pins():
    """If this fails, the parser found nothing — likely a path or format issue."""
    assert ENV_FILE.is_file(), f"environment.yml not found at {ENV_FILE}"
    assert PINS, (
        f"No '==' pins parsed from {ENV_FILE}. The parser may need updating, "
        f"or the file format has changed."
    )
