import sys
from pathlib import Path

# repo root = parent of "tests"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """
    Register custom markers so pytest doesn't emit PytestUnknownMarkWarning.

    Usage:
        @pytest.mark.integration   # tests that hit live external services
                                   # (AWS, NCEI, etc.) — skip with:
                                   #     pytest -m "not integration"
    """
    config.addinivalue_line(
        "markers",
        "integration: tests that hit live external services (network, S3, etc.); "
        "skip with `pytest -m 'not integration'`",
    )