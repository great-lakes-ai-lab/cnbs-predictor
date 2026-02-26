import sys
from pathlib import Path

# repo root = parent of "tests"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))