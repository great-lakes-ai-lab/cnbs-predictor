# tests/unit/test_paths.py
"""
Cross-platform path-safety tests for the ``src/`` library.

The goal is for this project to run on Windows, macOS, and Linux. Users
configure their working data path manually in production notebooks (a
deliberate, stable choice), so these tests focus on the *library* layer —
they check that nothing inside ``src/`` makes platform-specific assumptions
when handed a path.

Two flavors:

1. **Static tripwires** — scan source files for patterns that are usually
   bugs (hardcoded absolute paths for any one OS, manual ``+ "/"``
   concatenation). These fail if a future change reintroduces a known
   foot-gun.
2. **Behavioral checks** — call the path-taking utilities with both ``str``
   and ``pathlib.Path`` inputs and confirm they behave the same way, since
   ``pathlib.Path`` is the portable way to build paths.
"""

from pathlib import Path
import re

import pytest

from src.utilities import create_directory, get_files

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
SRC_FILES = sorted(SRC_DIR.glob("*.py"))


# ---------------------------------------------------------------------------
# Static tripwires
# ---------------------------------------------------------------------------
# Patterns that lock the code to a single OS. Each is a (regex, label) pair.
FORBIDDEN_PATTERNS = [
    (re.compile(r'["\']/Users/[^"\']+["\']'), "hardcoded macOS absolute path"),
    (re.compile(r'["\']/home/[^"\']+["\']'), "hardcoded Linux absolute path"),
    (re.compile(r'["\'][A-Za-z]:\\\\'),       "hardcoded Windows drive-letter path"),
    (re.compile(r'["\']/tmp/[^"\']+["\']'),   "hardcoded /tmp path (use tempfile / tmp_path)"),
    (re.compile(r'["\']/var/[^"\']+["\']'),   "hardcoded /var path"),
    (re.compile(r'["\']/opt/[^"\']+["\']'),   "hardcoded /opt path"),
]


@pytest.mark.parametrize(
    "src_file", SRC_FILES, ids=[p.name for p in SRC_FILES]
)
def test_no_hardcoded_absolute_paths(src_file):
    """
    No file in ``src/`` should contain an absolute path that only resolves
    on one OS. Accept paths as parameters or read them from config.
    """
    text = src_file.read_text()
    hits = []
    for pattern, label in FORBIDDEN_PATTERNS:
        for m in pattern.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append(f"  {src_file.name}:{line_no}  [{label}]  {m.group(0)!r}")
    assert not hits, "Forbidden absolute paths found:\n" + "\n".join(hits)


# Pattern for risky path concatenation: a string literal that's just a single
# path separator, used in a `+` concatenation. e.g. `dir + "/" + name`.
# This is portable-looking on the dev's machine but breaks the moment the
# other separator gets mixed in.
PATH_CONCAT = re.compile(r'\+\s*["\'][/\\]["\']\s*\+')


@pytest.mark.parametrize(
    "src_file", SRC_FILES, ids=[p.name for p in SRC_FILES]
)
def test_no_manual_path_separator_concatenation(src_file):
    """
    Building paths via ``dir + "/" + name`` (or ``"\\"``) is a portability
    hazard. Use ``os.path.join`` or ``pathlib.Path`` instead.
    """
    text = src_file.read_text()
    hits = []
    for m in PATH_CONCAT.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        hits.append(f"  {src_file.name}:{line_no}  {m.group(0)!r}")
    assert not hits, (
        "Manual path-separator concatenation found (use os.path.join or "
        "pathlib instead):\n" + "\n".join(hits)
    )


# ---------------------------------------------------------------------------
# Behavioral: key utilities should accept both str and pathlib.Path
# ---------------------------------------------------------------------------
class TestCreateDirectoryAcceptsBothStrAndPath:
    """
    ``create_directory`` is one of the most-called helpers in this project.
    It must accept both ``str`` and ``pathlib.Path`` so callers can use
    whichever is natural — and so the library stays portable.
    """

    def test_str_input_creates_directory(self, tmp_path):
        """Creates the directory when given a str path."""
        target = tmp_path / "from_str"
        create_directory(str(target))
        assert target.is_dir()

    def test_path_input_creates_directory(self, tmp_path):
        """Creates the directory when given a pathlib.Path."""
        target = tmp_path / "from_path"
        create_directory(target)  # pathlib.Path, not str
        assert target.is_dir()

    def test_str_and_path_inputs_are_equivalent(self, tmp_path):
        """str and Path inputs both create their directories equivalently."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        create_directory(str(a))
        create_directory(b)
        assert a.is_dir() and b.is_dir()


class TestGetFilesAcceptsBothStrAndPath:
    """
    ``get_files`` should accept both ``str`` and ``pathlib.Path`` for
    ``directory``. Note: ``get_files`` currently has a known bug where it
    always returns ``None`` — see test_utilities.py. These tests pin only
    that it does not *raise* on either input type.
    """

    def _populate(self, d: Path):
        """Create a couple of sample files in directory ``d``."""
        (d / "alpha.csv").write_text("x")
        (d / "beta.txt").write_text("x")

    def test_str_input_does_not_raise(self, tmp_path):
        """get_files accepts a str directory without raising."""
        self._populate(tmp_path)
        get_files(str(tmp_path), affix="suffix", identifier=".csv")

    def test_path_input_does_not_raise(self, tmp_path):
        """get_files accepts a pathlib.Path directory without raising."""
        self._populate(tmp_path)
        get_files(tmp_path, affix="suffix", identifier=".csv")
