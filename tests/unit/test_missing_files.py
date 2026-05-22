# tests/unit/test_missing_files.py
"""
Tests for clean error handling when required files are missing.

Issue #38 mentions checking that "required files exist (databases, mask
files, saved trained models)." That's a deployment-environment smoke check,
not a unit test. The unit-test angle is: when a required file *is* missing,
does the project fail loudly with a clear error rather than silently
producing garbage?

These tests cover that angle for the loader / model entry points. They
deliberately do NOT verify any production deployment paths exist.
"""

from pathlib import Path

import pytest

from src.hydro_utils import load_model
from src.data_loader import DataLoader
from src.data_processor import SeasonalCycleProcessor


# ===========================================================================
# load_model — both error branches in its documented contract
# ===========================================================================
class TestLoadModel:
    def test_unknown_model_name_raises_value_error(self):
        """If model_name isn't in models_info, raise ValueError with a clear message."""
        models_info = [{"model": "GP", "path": "/tmp/gp.joblib"}]
        with pytest.raises(ValueError, match="Model 'XYZ' not found"):
            load_model("XYZ", models_info)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """Documented behavior: FileNotFoundError if the .joblib path doesn't exist."""
        models_info = [{"model": "GP", "path": str(tmp_path / "does_not_exist.joblib")}]
        with pytest.raises(FileNotFoundError):
            load_model("GP", models_info)


# ===========================================================================
# SeasonalCycleProcessor.load — relies on pandas / open() to raise
# ===========================================================================
class TestSeasonalCycleProcessorLoad:
    def test_missing_climatology_raises_file_not_found(self, tmp_path):
        bad_path = str(tmp_path / "no_such_climatology.csv")
        meta_path = tmp_path / "meta.json"
        meta_path.write_text("{}")
        with pytest.raises(FileNotFoundError):
            SeasonalCycleProcessor.load(bad_path, str(meta_path))

    def test_missing_metadata_raises_file_not_found(self, tmp_path):
        # Create a real (empty) climatology so the first read succeeds far enough.
        clim = tmp_path / "clim.csv"
        clim.write_text("month,var_a\n1,0.0\n")
        bad_meta = str(tmp_path / "no_such_meta.json")
        with pytest.raises(FileNotFoundError):
            SeasonalCycleProcessor.load(str(clim), bad_meta)


# ===========================================================================
# DataLoader methods — each should fail clearly when files aren't present
# ===========================================================================
class TestDataLoaderMissingFiles:
    """
    Each loader uses pandas under the hood; pandas raises FileNotFoundError
    (a subclass of OSError) when its target doesn't exist. These tests pin
    that the loaders propagate that error rather than swallowing it.
    """

    def test_glcc_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader().glcc(str(tmp_path / "no_such_dir"))

    def test_l2swbm_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader().l2swbm(str(tmp_path / "no_such_dir"))

    def test_glsea_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader().glsea(str(tmp_path / "no_such_file.csv"))

    def test_lake_probabilities_missing_directory_raises(self, tmp_path):
        # Loader uses string concatenation; pass with trailing separator.
        with pytest.raises(FileNotFoundError):
            DataLoader().lake_probabilities(str(tmp_path / "no_such_dir") + "/")
