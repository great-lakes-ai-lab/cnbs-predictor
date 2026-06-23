# tests/unit/test_data_processor_pipeline.py
"""
End-to-end-ish unit tests for ``CFSProcessor.process_files``.

Strategy: the GRIB / NetCDF / database I/O is fully mocked out, but the
masking-math code path is exercised with real numpy + xarray arrays. This
pins the current pipeline contract: filename parsing, regridding,
mask-weighted averaging, and what gets written to the database.

Scope: precipitation path only. Temperature and evaporation paths follow
similar patterns and can be added once this baseline is in place.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from src.data_processor import CFSProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_fake_mask_dataset():
    """
    Build a tiny mask dataset: a 4x4 lat/lon grid where ``sup_lake`` is 1 in
    the central 2x2 block and 0 elsewhere.

    Returns
    -------
    SimpleNamespace
        A stand-in for ``netCDF4.Dataset``: anything that exposes
        ``.variables[name][:]`` will work since the production code uses
        exactly that pattern.
    """
    lat = np.array([46.5, 46.0, 45.5, 45.0])  # descending (matches GRIB convention)
    lon = np.array([-85.0, -84.5, -84.0, -83.5])  # ascending
    mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    return SimpleNamespace(
        variables={
            "latitude": lat,
            "longitude": lon,
            "sup_lake": mask,
        }
    )


def _build_fake_pgbf_dataset(pcp_value=0.001):
    """
    Build a fake xarray Dataset mimicking what ``cfgrib.open_dataset`` returns
    for a CFSv2 pgbf surface file: a single ``tp`` (total precipitation)
    DataArray on a lat/lon grid that fully encloses the mask grid.

    Latitude is descending to match GRIB convention so that the production
    ``slice(mask_lat.max(), mask_lat.min())`` selection works.
    """
    pcp_lat = np.linspace(47.0, 44.5, 6)  # descending
    pcp_lon = np.linspace(-85.5, -83.0, 6)  # ascending
    tp = xr.DataArray(
        np.full((len(pcp_lat), len(pcp_lon)), pcp_value),
        coords={"latitude": pcp_lat, "longitude": pcp_lon},
        dims=["latitude", "longitude"],
        name="tp",
    )
    return xr.Dataset({"tp": tp})


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestProcessFilesValidation:
    """
    These guardrails fire before any I/O — they exercise the input checks at
    the top of ``process_files``. The CFSDatabase is patched only to keep the
    constructor from touching a real SQLite file.
    """

    def _make_processor(self, monkeypatch):
        """Build a CFSProcessor with its database constructor mocked out."""
        monkeypatch.setattr("src.data_processor.CFSDatabase", lambda *a, **k: MagicMock())
        return CFSProcessor(database="fake.db", table="cfs")

    def test_missing_download_dir_raises(self, tmp_path, monkeypatch):
        """Raises ValueError when the download directory doesn't exist."""
        proc = self._make_processor(monkeypatch)
        with pytest.raises(ValueError, match="directory does not exist"):
            proc.process_files(
                str(tmp_path / "does_not_exist"),
                str(tmp_path / "ignored.nc"),
                ["sup_lake"],
            )

    def test_missing_mask_file_raises(self, tmp_path, monkeypatch):
        """Raises ValueError when the mask file doesn't exist."""
        proc = self._make_processor(monkeypatch)
        with pytest.raises(ValueError, match="mask_file not found"):
            proc.process_files(str(tmp_path), str(tmp_path / "missing.nc"), ["sup_lake"])

    def test_non_list_mask_variables_raises(self, tmp_path, monkeypatch):
        """Raises ValueError when mask_variables is not a list."""
        proc = self._make_processor(monkeypatch)
        # Create a real file so the mask_file check passes
        mask_path = tmp_path / "mask.nc"
        mask_path.write_text("")
        with pytest.raises(ValueError, match="mask_variables must be a list"):
            proc.process_files(str(tmp_path), str(mask_path), "sup_lake")


# ---------------------------------------------------------------------------
# Happy path — precipitation masking math
# ---------------------------------------------------------------------------
class TestPrecipitationMaskingMath:
    """
    With a constant precipitation field ``p`` over the lake mask, the math
    inside ``process_files`` reduces algebraically to::

        pcp_mm = (Σ(p · mask · area) · 4 · num_days) / Σ(mask · area)
               = p · 4 · num_days

    so for p = 0.001 and Jan 2024 (31 days) we expect 0.124. This pins the
    masking + regridding + filename parsing pipeline end-to-end.
    """

    def test_pgbf_january_writes_expected_row(self, tmp_path, monkeypatch):
        """A January pgbf file writes one DB row with the expected metadata and pcp_mm value."""
        # --- fakes ---
        fake_mask_ds = _build_fake_mask_dataset()
        pcp_value = 0.001
        fake_pgb_ds = _build_fake_pgbf_dataset(pcp_value=pcp_value)

        db_calls = []
        fake_db = MagicMock()
        fake_db.add = lambda *args: db_calls.append(args)

        monkeypatch.setattr("src.data_processor.nc.Dataset", lambda *a, **k: fake_mask_ds)
        monkeypatch.setattr(
            "src.data_processor.cfgrib.open_dataset",
            lambda *a, **k: fake_pgb_ds,
        )
        monkeypatch.setattr("src.data_processor.CFSDatabase", lambda *a, **k: fake_db)

        # --- file layout ---
        # Keep the mask file outside download_dir; otherwise process_files'
        # listdir loop would try to parse it as a GRIB filename.
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()
        # CFSv2 pgbf naming: pgbf.<member>.<cfs_run>.<forecast_yyyymm>.<...>.grib.grb2
        pgbf = "pgbf.01.2024010100.202401.avrg.grib.grb2"
        (download_dir / pgbf).write_text("")
        # Throw in an .idx file to confirm cleanup runs (non-fatal if it fails)
        (download_dir / (pgbf + ".idx")).write_text("")
        # process_files validates mask_file existence before nc.Dataset is called,
        # so the path must point to a real file even though its contents are unused.
        mask_path = tmp_path / "mask.nc"
        mask_path.write_text("")

        # --- run ---
        proc = CFSProcessor(database="fake.db", table="cfs")
        proc.process_files(str(download_dir), str(mask_path), ["sup_lake"])

        # --- assertions ---
        assert len(db_calls) == 1, f"expected 1 db row, got {len(db_calls)}: {db_calls}"
        cfs_run, year, month, lake, surface, var, value = db_calls[0]

        assert cfs_run == "2024010100"
        assert year == 2024
        assert month == 1
        assert lake == "superior"
        assert surface == "lake"
        assert var == "precipitation"

        # Math: constant pcp -> pcp_mm = p * 4 * num_days
        expected = pcp_value * 4 * 31
        assert value == pytest.approx(expected, rel=1e-9)

    def test_idx_files_are_removed(self, tmp_path, monkeypatch):
        """``process_files`` should strip stale ``.idx`` files before processing."""
        fake_mask_ds = _build_fake_mask_dataset()
        monkeypatch.setattr("src.data_processor.nc.Dataset", lambda *a, **k: fake_mask_ds)
        monkeypatch.setattr("src.data_processor.cfgrib.open_dataset", lambda *a, **k: _build_fake_pgbf_dataset())
        monkeypatch.setattr("src.data_processor.CFSDatabase", lambda *a, **k: MagicMock())

        download_dir = tmp_path / "downloads"
        download_dir.mkdir()
        idx = download_dir / "stale.idx"
        idx.write_text("")
        assert idx.exists()
        mask_path = tmp_path / "mask.nc"
        mask_path.write_text("")

        proc = CFSProcessor(database="fake.db", table="cfs")
        proc.process_files(str(download_dir), str(mask_path), ["sup_lake"])

        assert not idx.exists()

    def test_unknown_lake_prefix_is_silently_skipped(self, tmp_path, monkeypatch, capsys):
        """
        Sub-optimal current behavior: an unrecognized mask-variable prefix
        (e.g. ``foo_lake`` instead of ``sup_lake``) raises ``ValueError``
        inside the precipitation block, but the broad ``except Exception``
        swallows it — only a print statement reaches the caller and no row
        is written. This test pins that behavior so it's visible. A future
        revision should let the validation error propagate.
        """
        fake_mask_ds = SimpleNamespace(
            variables={
                "latitude": np.array([46.5, 46.0, 45.5, 45.0]),
                "longitude": np.array([-85.0, -84.5, -84.0, -83.5]),
                "foo_lake": np.ones((4, 4)),  # unknown prefix
            }
        )
        db_calls = []
        fake_db = MagicMock()
        fake_db.add = lambda *args: db_calls.append(args)

        monkeypatch.setattr("src.data_processor.nc.Dataset", lambda *a, **k: fake_mask_ds)
        monkeypatch.setattr("src.data_processor.cfgrib.open_dataset", lambda *a, **k: _build_fake_pgbf_dataset())
        monkeypatch.setattr("src.data_processor.CFSDatabase", lambda *a, **k: fake_db)

        download_dir = tmp_path / "downloads"
        download_dir.mkdir()
        (download_dir / "pgbf.01.2024010100.202401.avrg.grib.grb2").write_text("")
        mask_path = tmp_path / "mask.nc"
        mask_path.write_text("")

        proc = CFSProcessor(database="fake.db", table="cfs")
        proc.process_files(str(download_dir), str(mask_path), ["foo_lake"])

        # No DB row should have been written.
        assert db_calls == []
        # The error message should have been printed (current behavior).
        captured = capsys.readouterr()
        assert "ERROR processing precipitation data" in captured.out
