# %%
import os
from pathlib import Path

import pandas as pd
import xarray as xr
import cfgrib


# %%
def parse_flx_filename(filename):
    """
    Parse:
        flxf.01.YYYYMMDDHH.YYYYMM.avrg.grib.grb2

    Returns:
        init_time
        valid_time
    """

    parts = Path(filename).name.split(".")

    init_time = pd.to_datetime(
        parts[2],
        format="%Y%m%d%H"
    )

    valid_time = pd.to_datetime(
        parts[3],
        format="%Y%m"
    )

    return init_time, valid_time


# %%
def read_cfs_variable(
    file,
    filter_by_keys,
    variable_name,
    output_name=None,
):
    """
    Read a variable from a CFS GRIB2 file.
    """

    init_time, valid_time = parse_flx_filename(file)

    ds = cfgrib.open_dataset(
        file,
        engine="cfgrib",
        filter_by_keys=filter_by_keys,
        decode_timedelta=False,
    )

    # CFS variable name
    da = ds[variable_name]

    # Rename to something cleaner if output_name is provided
    if output_name is not None:
        da = da.rename(output_name)

    da = da.expand_dims(
        cfs_run_time=[init_time],
        valid_time=[valid_time],
    )

    da = da.drop_vars(
        ["time", "step", "heightAboveGround"],
        errors="ignore",
    )

    return da


# %%
def read_cfs_2m_temperature(file):
    """
    Read 2-m air temperature from a CFS GRIB2 file.

    Tries:
        1. avg_2t
        2. mean2t

    If neither variable is found, returns None.
    """

    init_time, valid_time = parse_flx_filename(file)

    try:

        ds = cfgrib.open_dataset(
            file,
            engine="cfgrib",
            filter_by_keys={
                "typeOfLevel": "heightAboveGround",
                "level": 2,
            },
            decode_timedelta=False,
        )

    except Exception as e:

        print(
            f"        Could not open {Path(file).name}: {e}"
        )

        return None

    # -------------------------------------------------
    # Try avg_2t first, then mean2t
    # -------------------------------------------------

    if "avg_2t" in ds.data_vars:

        variable_name = "avg_2t"

    elif "mean2t" in ds.data_vars:

        variable_name = "mean2t"

    else:

        print(
            f"        Skipping {Path(file).name}: "
            "avg_2t and mean2t not found"
        )

        ds.close()

        return None

    print(
        f"        Using {variable_name} "
        f"from {Path(file).name}"
    )

    da = ds[variable_name]

    # Use a consistent output variable name regardless
    # of which CFS variable was found.
    da = da.rename("avg_2t")

    da = da.expand_dims(
        cfs_run_time=[init_time],
        valid_time=[valid_time],
    )

    da = da.drop_vars(
        ["time", "step", "heightAboveGround"],
        errors="ignore",
    )

    # Load into memory before closing the GRIB dataset
    da = da.load()

    ds.close()

    return da


# %%
def remove_idx_files(directory):
    """
    Remove cfgrib index files (*.idx) from a directory.
    """

    directory = Path(directory)

    for file in directory.glob("*.idx"):

        try:

            file.unlink()

            print(
                f"    Removed {file.name}"
            )

        except Exception as e:

            print(
                f"    Could not remove {file.name}: {e}"
            )


# %%
def build_cfs_temperature_archive(
    input_directory,
    output_zarr,
):

    input_directory = Path(input_directory)

    # -------------------------------------------------
    # Precompute all valid forecast months
    #
    # This only reads filenames; it does not open GRIB
    # files or load any data.
    # -------------------------------------------------

    all_files = sorted(
        input_directory.rglob("flxf*.grb2")
    )

    all_valid_times = sorted(
        {
            parse_flx_filename(str(file))[1]
            for file in all_files
        }
    )

    print(
        f"Found {len(all_valid_times)} valid forecast months."
    )

    first = True

    # -------------------------------------------------
    # Find initialization directories
    #
    # Sort numerically based on YYYYMMDD directory name
    # -------------------------------------------------

    init_directories = sorted(
        [
            d
            for d in input_directory.iterdir()
            if d.is_dir()
            and d.name.isdigit()
        ],
        key=lambda d: int(d.name),
    )

    # -------------------------------------------------
    # Process each initialization directory
    # -------------------------------------------------

    for init_directory in init_directories:

        files = sorted(
            init_directory.glob("flxf*.grb2")
        )

        if len(files) == 0:
            continue

        print(
            f"\nReading in {init_directory.name}"
        )

        print(
            f"    Found {len(files)} files"
        )

        # -------------------------------------------------
        # Group files by initialization time
        # -------------------------------------------------

        grouped = {}

        for file in files:

            init_time, _ = parse_flx_filename(
                str(file)
            )

            grouped.setdefault(
                init_time,
                []
            ).append(str(file))

        # -------------------------------------------------
        # Process each CFS initialization cycle
        # -------------------------------------------------

        for init_time in sorted(grouped):

            print(
                f"    Processing {init_time}"
            )

            monthly = []

            # ---------------------------------------------
            # Read each valid month
            # ---------------------------------------------

            for file in sorted(
                grouped[init_time]
            ):

                da = read_cfs_2m_temperature(
                    file
                )

                # Skip files where neither avg_2t nor
                # mean2t was found
                if da is None:
                    continue

                monthly.append(da)

            # ---------------------------------------------
            # If no valid files were found for this run,
            # skip it rather than crashing xr.concat()
            # ---------------------------------------------

            if len(monthly) == 0:

                print(
                    f"    No valid temperature files "
                    f"found for {init_time}. Skipping."
                )

                continue

            # ---------------------------------------------
            # Combine forecast months
            # ---------------------------------------------

            run = xr.concat(
                monthly,
                dim="valid_time",
            )

            # ---------------------------------------------
            # Ensure identical valid_time dimension
            # ---------------------------------------------

            run = run.reindex(
                valid_time=all_valid_times
            )

            ds = run.to_dataset()

            # ---------------------------------------------
            # First write
            # ---------------------------------------------

            if first:

                lat_size = ds.sizes["latitude"]
                lon_size = ds.sizes["longitude"]

                encoding = {
                    "avg_2t": {
                        "chunks": (
                            1,
                            len(all_valid_times),
                            lat_size,
                            lon_size,
                        )
                    },
                    "cfs_run_time": {
                        "units": (
                            "hours since "
                            "1970-01-01 00:00:00"
                        ),
                    },
                    "valid_time": {
                        "units": (
                            "days since "
                            "1970-01-01 00:00:00"
                        ),
                    },
                }

                ds.to_zarr(
                    output_zarr,
                    mode="w",
                    consolidated=True,
                    encoding=encoding,
                    zarr_format=2,
                )

                first = False

            # ---------------------------------------------
            # Append subsequent initialization times
            # ---------------------------------------------

            else:

                ds.to_zarr(
                    output_zarr,
                    mode="a",
                    append_dim="cfs_run_time",
                    zarr_format=2,
                )

            # ---------------------------------------------
            # Free memory
            # ---------------------------------------------

            del monthly
            del run
            del ds

            print(
                "        written"
            )

        # -------------------------------------------------
        # Remove cfgrib .idx files before moving on
        # to the next initialization directory
        # -------------------------------------------------

        remove_idx_files(
            init_directory
        )

    print("\nFinished.")


# %%
build_cfs_temperature_archive(
    input_directory=(
        "/Users/ljob/Desktop/"
        "cnbs-predictor/data/cfs"
    ),
    output_zarr=(
        "/Users/ljob/Desktop/"
        "cnbs-predictor/data/zarr/"
        "cfs_2m_temperature.zarr"
    ),
)


# %%
# Check the resulting Zarr file

zarr_path = (
    "/Users/ljob/Desktop/"
    "cnbs-predictor/data/zarr/"
    "cfs_2m_temperature.zarr"
)

ds = xr.open_zarr(
    zarr_path,
    consolidated=True,
)

print(ds)