"""Shared pytest configuration: makes the repo root importable for tests.

Custom markers (``network``, ``integration``) are registered centrally in
``pyproject.toml`` under ``[tool.pytest.ini_options]``.
"""

import sys
from pathlib import Path

# repo root = parent of "tests"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))