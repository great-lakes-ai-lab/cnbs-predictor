"""Visualization functions for Great Lakes NBS forecast output.

Produces the figures used to inspect and communicate forecast results:

- :func:`plot_cnbs_forecast` — static 4x4 grid (lake x component) of the
  precipitation, evaporation, runoff, and NBS forecasts with 95% ranges
- :func:`plot_cnbs_forecast_interactive` — the same grid as an interactive
  Plotly figure saved to HTML
- :func:`plot_nbs_forecast` — per-lake NBS forecast with climatology overlay
- :func:`plot_cep_timeseries` — Climatology Exceedance Probability (CEP)
  timeseries per lake
- :func:`plot_cep_spatial` — month-by-month spatial CEP maps over the basin

Each function optionally saves to ``filename`` and otherwise displays the
figure. The plotting backends (matplotlib, plotly, cartopy) are mocked when
building the documentation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.patches as mpatches
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore", message="This axis already has a converter set")

def plot_cnbs_forecast(
        df,
        filename=None,
        units="mm",
        date_col="forecast_month"
    ):
    """
    Static Matplotlib plot for Great Lakes CNBS forecasts.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe containing:
        ["cfs_run", "forecast_month", "model", "lake", "component",
         "value [mm]", "value [cms]"]

        Alternatively, the forecast month can be represented by
        separate "year" and "month" columns.

    filename : str, optional
        Path to save the PNG. If None, the plot is displayed only
        and not saved.

    units : str, default="mm"
        Units to plot. Options:
        - "mm"  : uses "value [mm]"
        - "cms" : uses "value [cms]"

    date_col : str or list, default="forecast_month"
        Column(s) used to determine the forecast date.

        Options:
        - "forecast_month": A single column in YYYY-MM format.
        - ["year", "month"]&#58; Separate columns where year is YYYY
          and month is an integer from 1-12.
    """

    df = df.copy()

    # --------------------------------------------------------------
    # Validate date columns
    # --------------------------------------------------------------
    if date_col == "forecast_month":
        required_date_cols = {"forecast_month"}

    elif date_col == ["year", "month"]:
        required_date_cols = {"year", "month"}

    else:
        raise ValueError(
            "date_col must be either 'forecast_month' or ['year', 'month']"
        )

    required_cols = (
        required_date_cols
        | {
            "model",
            "lake",
            "component",
            "value [mm]",
            "value [cms]"
        }
    )

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataframe is missing required columns: {missing}"
        )

    # --------------------------------------------------------------
    # Select value column
    # --------------------------------------------------------------
    if units.lower() == "mm":
        value_col = "value [mm]"
        units_label = "mm"

    elif units.lower() == "cms":
        value_col = "value [cms]"
        units_label = "cms"

    else:
        raise ValueError(
            "units must be either 'mm' or 'cms'"
        )

    # --------------------------------------------------------------
    # Build datetime column for plotting
    # --------------------------------------------------------------
    if date_col == "forecast_month":

        df["date"] = pd.to_datetime(
            df["forecast_month"],
            format="%Y-%m"
        )

    else:

        df["date"] = pd.to_datetime(
            dict(
                year=df["year"].astype(int),
                month=df["month"].astype(int),
                day=1
            )
        )

        # Create standardized forecast_month for grouping
        df["forecast_month"] = (
            df["date"].dt.strftime("%Y-%m")
        )

    # --------------------------------------------------------------
    # Unique models
    # --------------------------------------------------------------
    models_info = df["model"].unique().tolist()

    custom_colors = [
        "blue",
        "green",
        "red",
        "purple",
        "orange",
        "brown",
        "pink",
        "olive",
        "cyan",
        "magenta"
    ]

    if len(models_info) > len(custom_colors):
        raise ValueError(
            f"Number of models ({len(models_info)}) exceeds "
            f"available colors ({len(custom_colors)}). Add more colors."
        )

    model_colors = {
        model: custom_colors[i]
        for i, model in enumerate(models_info)
    }

    # --------------------------------------------------------------
    # Lakes and components
    # --------------------------------------------------------------
    lakes = [
        "superior",
        "michigan-huron",
        "erie",
        "ontario"
    ]

    components = [
        "precipitation",
        "evaporation",
        "runoff",
        "nbs"
    ]

    # --------------------------------------------------------------
    # Create 4x4 subplot grid
    # --------------------------------------------------------------
    fig, axs = plt.subplots(
        4,
        4,
        figsize=(18, 8)
    )

    shared_y_min = float("inf")
    shared_y_max = float("-inf")

    col4_y_min = float("inf")
    col4_y_max = float("-inf")

    # --------------------------------------------------------------
    # Plot each lake/component combination
    # --------------------------------------------------------------
    for row, lake in enumerate(lakes):

        for col, comp in enumerate(components):

            ax = axs[row, col]

            lake_df = df[
                (df["lake"] == lake) &
                (df["component"] == comp)
            ]

            if lake_df.empty:
                continue

            # ------------------------------------------------------
            # Confidence intervals
            # ------------------------------------------------------
            lower = (
                lake_df
                .groupby("forecast_month")[value_col]
                .quantile(0.025)
                .reset_index()
            )

            upper = (
                lake_df
                .groupby("forecast_month")[value_col]
                .quantile(0.975)
                .reset_index()
            )

            lower["date"] = pd.to_datetime(
                lower["forecast_month"],
                format="%Y-%m"
            )

            upper["date"] = pd.to_datetime(
                upper["forecast_month"],
                format="%Y-%m"
            )

            # ------------------------------------------------------
            # Plot each model
            # ------------------------------------------------------
            for model_name in models_info:

                model_df = lake_df[
                    lake_df["model"] == model_name
                ]

                if model_df.empty:
                    continue

                mean_df = (
                    model_df
                    .groupby("forecast_month")[value_col]
                    .mean()
                    .reset_index()
                )

                mean_df["date"] = pd.to_datetime(
                    mean_df["forecast_month"],
                    format="%Y-%m"
                )

                ax.plot(
                    mean_df["date"],
                    mean_df[value_col],
                    marker="o",
                    markersize=3,
                    color=model_colors[model_name],
                    label=model_name
                )

            # ------------------------------------------------------
            # 95% confidence interval
            # ------------------------------------------------------
            ax.fill_between(
                lower["date"],
                lower[value_col],
                upper[value_col],
                color="gray",
                alpha=0.2,
                label="95%"
            )

            ax.axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=1
            )

            ax.grid(
                True,
                linestyle="--",
                alpha=0.6
            )

            # ------------------------------------------------------
            # Track y limits
            # ------------------------------------------------------
            current_y_min = lower[value_col].min()
            current_y_max = upper[value_col].max()

            if col < 3:
                shared_y_min = min(
                    shared_y_min,
                    current_y_min
                )
                shared_y_max = max(
                    shared_y_max,
                    current_y_max
                )

            else:
                col4_y_min = min(
                    col4_y_min,
                    current_y_min
                )
                col4_y_max = max(
                    col4_y_max,
                    current_y_max
                )

            # ------------------------------------------------------
            # Titles
            # ------------------------------------------------------
            if row == 0:
                ax.set_title(
                    [
                        "Precipitation",
                        "Evaporation",
                        "Runoff",
                        "NBS"
                    ][col],
                    fontsize=14
                )

            # ------------------------------------------------------
            # X-axis
            # ------------------------------------------------------
            if row == 3:

                ax.xaxis.set_major_formatter(
                    mdates.DateFormatter("%m-%Y")
                )

                ax.tick_params(
                    axis="x",
                    rotation=45
                )

                for label in ax.get_xticklabels():
                    label.set_horizontalalignment("right")

            else:
                ax.set_xticklabels([])

            # ------------------------------------------------------
            # Y-axis
            # ------------------------------------------------------
            if col == 0:
                ax.set_ylabel(
                    lake.capitalize(),
                    fontsize=14
                )
            else:
                ax.set_yticklabels([])

    # --------------------------------------------------------------
    # Apply y-axis limits
    # --------------------------------------------------------------
    for row_idx in range(4):

        for col_idx in range(4):

            if col_idx < 3:

                axs[row_idx, col_idx].set_ylim(
                    shared_y_min - 10,
                    shared_y_max + 10
                )

            else:

                axs[row_idx, col_idx].set_ylim(
                    col4_y_min - 10,
                    col4_y_max + 10
                )

            axs[row_idx, col_idx].set_xlim(
                df["date"].min(),
                df["date"].max()
            )

    # --------------------------------------------------------------
    # Figure title
    # --------------------------------------------------------------
    fig.suptitle(
        f"Great Lakes 12-Month CNBS Forecast [{units_label}]",
        fontsize=16
    )

    axs[0, 3].legend(
        loc="lower left",
        bbox_to_anchor=(1, 0.04),
        fontsize=12
    )

    plt.tight_layout()

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------
    if filename:

        os.makedirs(
            os.path.dirname(filename),
            exist_ok=True
        )

        plt.savefig(
            filename,
            bbox_inches="tight"
        )

    plt.show()


def plot_cnbs_forecast_interactive(
        df,
        filename=None,
        units="mm",
        date_col="forecast_month"
    ):
    """
    Interactive Plotly plot for Great Lakes CNBS forecasts.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe containing:
        ["cfs_run", "forecast_month", "model", "lake", "component",
         "value [mm]", "value [cms]"]

        Alternatively, the forecast month can be represented by
        separate "year" and "month" columns.

    filename : str, optional
        File path for saving the interactive HTML. Defaults to None.

    units : str, default="mm"
        Units to plot:
        - "mm"  : uses "value [mm]"
        - "cms" : uses "value [cms]"

    date_col : str or list, default="forecast_month"
        Column(s) used to determine the forecast date.

        Options:
        - "forecast_month": A single column in YYYY-MM format.
        - ["year", "month"]&#58; Separate columns where year is YYYY
          and month is an integer from 1-12.
    """

    df = df.copy()

    # --------------------------------------------------------------
    # Validate date columns
    # --------------------------------------------------------------
    if date_col == "forecast_month":
        required_date_cols = {"forecast_month"}

    elif date_col == ["year", "month"]:
        required_date_cols = {"year", "month"}

    else:
        raise ValueError(
            "date_col must be either 'forecast_month' or ['year', 'month']"
        )

    required_cols = (
        required_date_cols
        | {
            "model",
            "lake",
            "component",
            "value [mm]",
            "value [cms]"
        }
    )

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataframe is missing required columns: {missing}"
        )

    # --------------------------------------------------------------
    # Select units
    # --------------------------------------------------------------
    if units.lower() == "mm":
        value_col = "value [mm]"
        units_label = "mm"

    elif units.lower() == "cms":
        value_col = "value [cms]"
        units_label = "cms"

    else:
        raise ValueError(
            "units must be either 'mm' or 'cms'"
        )

    # --------------------------------------------------------------
    # Build datetime column
    # --------------------------------------------------------------
    if date_col == "forecast_month":

        df["date"] = pd.to_datetime(
            df["forecast_month"],
            format="%Y-%m"
        )

    else:

        df["date"] = pd.to_datetime(
            dict(
                year=df["year"].astype(int),
                month=df["month"].astype(int),
                day=1
            )
        )

        # Standardize to YYYY-MM for grouping
        df["forecast_month"] = (
            df["date"].dt.strftime("%Y-%m")
        )

    # --------------------------------------------------------------
    # Models
    # --------------------------------------------------------------
    models_info = df["model"].unique().tolist()

    custom_colors = [
        "blue",
        "green",
        "red",
        "purple",
        "orange",
        "brown",
        "pink",
        "olive",
        "cyan",
        "magenta"
    ]

    if len(models_info) > len(custom_colors):
        raise ValueError(
            f"Number of models ({len(models_info)}) exceeds "
            f"available colors ({len(custom_colors)})."
        )

    model_colors = {
        model: custom_colors[i]
        for i, model in enumerate(models_info)
    }

    # --------------------------------------------------------------
    # Lakes and components
    # --------------------------------------------------------------
    lakes = [
        "superior",
        "michigan-huron",
        "erie",
        "ontario"
    ]

    components = [
        "precipitation",
        "evaporation",
        "runoff",
        "nbs"
    ]

    # --------------------------------------------------------------
    # Create 4x4 subplot grid
    # --------------------------------------------------------------
    fig = make_subplots(
        rows=4,
        cols=4,
        subplot_titles=[
            "Precipitation",
            "Evaporation",
            "Runoff",
            "NBS"
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.008
    )

    global_y_min = float("inf")
    global_y_max = float("-inf")

    # --------------------------------------------------------------
    # Plot each lake/component
    # --------------------------------------------------------------
    for row, lake in enumerate(lakes):

        for col, comp in enumerate(components):

            lake_df = df[
                (df["lake"] == lake) &
                (df["component"] == comp)
            ]

            if lake_df.empty:
                continue

            # ------------------------------------------------------
            # 95% confidence bounds
            # ------------------------------------------------------
            lower = (
                lake_df
                .groupby("forecast_month")[value_col]
                .quantile(0.025)
                .reset_index()
            )

            upper = (
                lake_df
                .groupby("forecast_month")[value_col]
                .quantile(0.975)
                .reset_index()
            )

            lower["date"] = pd.to_datetime(
                lower["forecast_month"],
                format="%Y-%m"
            )

            upper["date"] = pd.to_datetime(
                upper["forecast_month"],
                format="%Y-%m"
            )

            # ------------------------------------------------------
            # Plot each model
            # ------------------------------------------------------
            for model_name in models_info:

                model_df = lake_df[
                    lake_df["model"] == model_name
                ]

                if model_df.empty:
                    continue

                mean_df = (
                    model_df
                    .groupby("forecast_month")[value_col]
                    .mean()
                    .reset_index()
                )

                mean_df["date"] = pd.to_datetime(
                    mean_df["forecast_month"],
                    format="%Y-%m"
                )

                # --------------------------------------------------
                # Confidence interval shading
                # --------------------------------------------------
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat([
                            mean_df["date"],
                            mean_df["date"][::-1]
                        ]),
                        y=pd.concat([
                            lower[value_col],
                            upper[value_col][::-1]
                        ]),
                        fill="toself",
                        fillcolor="rgba(169,169,169,0.2)",
                        line=dict(
                            color="rgba(255,255,255,0)"
                        ),
                        showlegend=False,
                        hovertemplate=(
                            "Date: %{x}<br>"
                            "95% interval<br>"
                            f"{units_label}: %{{y}}"
                        )
                    ),
                    row=row + 1,
                    col=col + 1
                )

                # --------------------------------------------------
                # Mean model line
                # --------------------------------------------------
                fig.add_trace(
                    go.Scatter(
                        x=mean_df["date"],
                        y=mean_df[value_col],
                        mode="lines+markers",
                        name=model_name,
                        line=dict(
                            shape="linear",
                            width=2,
                            color=model_colors[model_name]
                        ),
                        marker=dict(size=4),
                        showlegend=(
                            row == 0 and col == 0
                        ),
                        hovertemplate=(
                            "Date: %{x}<br>"
                            f"{units_label}: %{{y}}<br>"
                            "Model: " + model_name
                        )
                    ),
                    row=row + 1,
                    col=col + 1
                )

                global_y_min = min(
                    global_y_min,
                    lower[value_col].min()
                )

                global_y_max = max(
                    global_y_max,
                    upper[value_col].max()
                )

            # ------------------------------------------------------
            # Zero reference line
            # ------------------------------------------------------
            fig.add_trace(
                go.Scatter(
                    x=[
                        lake_df["date"].min(),
                        lake_df["date"].max()
                    ],
                    y=[0, 0],
                    mode="lines",
                    line=dict(
                        color="black",
                        dash="dash"
                    ),
                    showlegend=False
                ),
                row=row + 1,
                col=col + 1
            )

            # ------------------------------------------------------
            # Axis formatting
            # ------------------------------------------------------
            fig.update_xaxes(
                tickformat="%m-%Y",
                tickangle=-45,
                showticklabels=(row == 3),
                showline=True,
                linewidth=2,
                linecolor="black",
                row=row + 1,
                col=col + 1
            )

            fig.update_yaxes(
                title_text=lake.title() if col == 0 else None,
                showline=True,
                linewidth=2,
                linecolor="black",
                row=row + 1,
                col=col + 1
            )

    # --------------------------------------------------------------
    # Consistent y-axis limits
    # --------------------------------------------------------------
    for r in range(1, 5):

        for c in range(1, 5):

            fig.update_yaxes(
                range=[
                    global_y_min - 10,
                    global_y_max + 10
                ],
                row=r,
                col=c
            )

    # --------------------------------------------------------------
    # Gridlines
    # --------------------------------------------------------------
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor="LightGray"
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor="LightGray"
    )

    # --------------------------------------------------------------
    # Layout
    # --------------------------------------------------------------
    fig.update_layout(
        title_text=(
            f"Great Lakes 12-Month CNBS "
            f"[{units_label}] Forecast"
        ),
        title_x=0.5,
        height=800,
        width=1500,
        showlegend=True,
        legend=dict(
            x=1.01,
            y=1.01,
            traceorder="normal",
            orientation="v",
            title="Model"
        ),
        plot_bgcolor="white"
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------
    if filename:

        os.makedirs(
            os.path.dirname(filename),
            exist_ok=True
        )

        fig.write_html(filename)

    fig.show()

def plot_nbs_forecast(
    df_filtered,
    prob,
    filename=None,
    units="mm",
    date_col="forecast_month"
):
    """
    Plot NBS forecasts for the Great Lakes with ensemble model lines,
    95% range, and climatology.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Long-format forecast data containing:
        'forecast_month' OR 'year' and 'month',
        'lake', 'model', 'component',
        'value [mm]', and 'value [cms]'.

    prob : pd.DataFrame
        Probability of exceedance data containing:
        'date', 'lake', 'prob_exceedance', and 'value'.

    filename : str, optional
        Path to save the PNG.

    units : str, default="mm"
        Units to plot:
        - "mm"  uses 'value [mm]'
        - "cms" uses 'value [cms]'

    date_col : str, default="forecast_month"
        Specifies how the forecast date is provided.

        - "forecast_month": Uses a 'forecast_month' column in YYYY-MM format.
        - "year_month": Uses separate 'year' and 'month' columns.
    """

    # ---------------------------------------------------------
    # Select units
    # ---------------------------------------------------------
    if units.lower() == "mm":
        value_col = "value [mm]"
        units_label = "mm"
    elif units.lower() == "cms":
        value_col = "value [cms]"
        units_label = "cms"
    else:
        raise ValueError("units must be either 'mm' or 'cms'")

    # ---------------------------------------------------------
    # Copy dataframe
    # ---------------------------------------------------------
    df = df_filtered.copy()

    # ---------------------------------------------------------
    # Validate and create date column
    # ---------------------------------------------------------
    if date_col == "forecast_month":

        df["date"] = pd.to_datetime(
            df["forecast_month"],
            format="%Y-%m"
        )

    else:

        df["date"] = pd.to_datetime(
            dict(
                year=df["year"].astype(int),
                month=df["month"].astype(int),
                day=1
            )
        )

        # Standardize to YYYY-MM for grouping
        df["forecast_month"] = (
            df["date"].dt.strftime("%Y-%m")
        )


    # ---------------------------------------------------------
    # Only plot NBS
    # ---------------------------------------------------------
    df = df[
        df["component"].str.lower() == "nbs"
    ].copy()

    if df.empty:
        raise ValueError(
            "No NBS data found in dataframe."
        )

    # ---------------------------------------------------------
    # Unique models
    # ---------------------------------------------------------
    models_info = df["model"].unique().tolist()

    custom_colors = [
        "blue", "green", "red", "purple", "orange",
        "brown", "pink", "olive", "cyan", "magenta"
    ]

    if len(models_info) > len(custom_colors):
        raise ValueError(
            f"Number of models ({len(models_info)}) exceeds "
            f"available colors ({len(custom_colors)}). "
            "Add more colors."
        )

    model_colors = {
        model: custom_colors[i]
        for i, model in enumerate(models_info)
    }

    # ---------------------------------------------------------
    # Lakes
    # ---------------------------------------------------------
    lakes = [
        "superior",
        "michigan-huron",
        "erie",
        "ontario"
    ]

    # ---------------------------------------------------------
    # Begin plotting
    # ---------------------------------------------------------
    fig, axs = plt.subplots(
        len(lakes),
        1,
        figsize=(10, 12),
        squeeze=False
    )

    fig.suptitle(
        f"Great Lakes 12-Month NBS [{units_label}] Forecast",
        fontsize=16,
        x=0.2,
        ha="left"
    )

    shared_y_min = float("inf")
    shared_y_max = float("-inf")

    for row, lake in enumerate(lakes):

        ax = axs[row, 0]

        lake_df = df[
            df["lake"] == lake
        ]

        if lake_df.empty:
            continue

        # -----------------------------------------------------
        # Confidence intervals
        # -----------------------------------------------------
        lower = (
            lake_df
            .groupby("forecast_month")[value_col]
            .quantile(0.025)
            .reset_index()
        )

        upper = (
            lake_df
            .groupby("forecast_month")[value_col]
            .quantile(0.975)
            .reset_index()
        )

        lower["date"] = pd.to_datetime(
            lower["forecast_month"],
            format="%Y-%m"
        )

        upper["date"] = pd.to_datetime(
            upper["forecast_month"],
            format="%Y-%m"
        )

        # -----------------------------------------------------
        # Plot each model
        # -----------------------------------------------------
        for model_name in models_info:

            model_df = lake_df[
                lake_df["model"] == model_name
            ]

            if model_df.empty:
                continue

            mean_df = (
                model_df
                .groupby("forecast_month")[value_col]
                .mean()
                .reset_index()
            )

            mean_df["date"] = pd.to_datetime(
                mean_df["forecast_month"],
                format="%Y-%m"
            )

            ax.plot(
                mean_df["date"],
                mean_df[value_col],
                marker="o",
                markersize=3,
                color=model_colors[model_name],
                label=model_name
            )

        # -----------------------------------------------------
        # 95% range
        # -----------------------------------------------------
        ax.fill_between(
            lower["date"],
            lower[value_col],
            upper[value_col],
            color="gray",
            alpha=0.2,
            label="95%"
        )

        # -----------------------------------------------------
        # Climatology (PoE = 0.5)
        # -----------------------------------------------------
        lake_prob = prob.query(
            f"lake == '{lake}' and prob_exceedance == 0.5"
        )

        if not lake_prob.empty:

            lake_prob = lake_prob.copy()

            lake_prob["date"] = pd.to_datetime(
                lake_prob["date"]
            )

            ax.plot(
                lake_prob["date"],
                lake_prob["value"],
                color="black",
                linestyle="--",
                linewidth=2,
                label="Climatology"
            )

        # -----------------------------------------------------
        # Formatting
        # -----------------------------------------------------
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.6
        )

        ax.set_xlabel("")

        # Track y limits
        shared_y_min = min(
            shared_y_min,
            lower[value_col].min()
        )

        shared_y_max = max(
            shared_y_max,
            upper[value_col].max()
        )

        # X-axis formatting
        if row == len(lakes) - 1:

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%m-%Y")
            )

            ax.tick_params(
                axis="x",
                rotation=45
            )

            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")

        else:
            ax.set_xticklabels([])

        ax.set_ylabel(
            lake.title(),
            fontsize=14
        )

    # ---------------------------------------------------------
    # Apply common y-axis
    # ---------------------------------------------------------
    for ax in axs[:, 0]:

        ax.set_ylim(
            shared_y_min - 10,
            shared_y_max + 10
        )

        ax.set_xlim(
            df["date"].min(),
            df["date"].max()
        )

    # ---------------------------------------------------------
    # Legend only on first subplot
    # ---------------------------------------------------------
    axs[0, 0].legend(
        loc="lower left",
        bbox_to_anchor=(1, 0.3),
        fontsize=12
    )

    for r in range(1, len(lakes)):

        if axs[r, 0].get_legend() is not None:
            axs[r, 0].legend_.remove()

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    if filename:

        os.makedirs(
            os.path.dirname(filename),
            exist_ok=True
        )

        plt.savefig(
            filename,
            bbox_inches="tight"
        )

    plt.show()

def plot_cep_timeseries(df_cep, filename=None):
    """
    Plot Great Lakes 12-Month mean CEP timeseries for multiple models.

    Parameters
    ----------
    df_cep : pd.DataFrame
        CEP dataframe. Must contain:
            - date
            - lake
            - model
            - cep

    filename : str, optional
        Path to save figure.
    """

    df = df_cep.copy()

    df["forecast_month"] = pd.to_datetime(df["forecast_month"], format="%Y-%m")
    df["lake"] = df["lake"].str.lower().str.strip()

    lakes = ["superior", "michigan-huron", "erie", "ontario"]

    lake_labels = {
        "superior": "Superior",
        "michigan-huron": "Mich-Huron",
        "erie": "Erie",
        "ontario": "Ontario",
    }

    models_info = sorted(df["model"].unique().tolist())

    custom_colors = [
        "blue", "green", "red", "purple",
        "orange", "brown", "pink", "olive",
        "cyan", "magenta"
    ]

    if len(models_info) > len(custom_colors):
        raise ValueError(
            f"Number of models ({len(models_info)}) exceeds "
            f"available colors ({len(custom_colors)})."
        )

    model_colors = {
        model: custom_colors[i]
        for i, model in enumerate(models_info)
    }

    fig, axs = plt.subplots(
        len(lakes),
        1,
        figsize=(10, 12),
        squeeze=False
    )

    fig.suptitle(
        "Great Lakes 12-Month Mean Climatology Exceedance Probability (NBS)",
        fontsize=16,
        x=0.17,
        ha="left"
    )

    for row, lake in enumerate(lakes):
        ax = axs[row, 0]

        lake_df = df[df["lake"] == lake].copy()

        for model_name in models_info:
            model_df = (
                lake_df[lake_df["model"] == model_name]
                .sort_values("forecast_month")
            )

            if model_df.empty:
                continue

            ax.plot(
                model_df["forecast_month"],
                model_df["cep"],
                marker="o",
                markersize=4,
                linewidth=2,
                color=model_colors[model_name],
                label=model_name
            )

        ax.axhline(
            0.5,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7
        )

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_xlim(df["forecast_month"].min(), df["forecast_month"].max())
        ax.set_ylabel(lake_labels[lake], fontsize=14)

        if row == len(lakes) - 1:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
            ax.tick_params(axis="x", rotation=45)

            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
        else:
            ax.set_xticklabels([])

    axs[0, 0].legend(
        loc="lower left",
        bbox_to_anchor=(1, 0.525),
        fontsize=12
    )

    plt.tight_layout()

    if filename:
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)

        plt.savefig(filename, bbox_inches="tight", dpi=300)

    plt.show()

def plot_cep_spatial(
    df_cep,
    model=None,
    value_col="cep",
    filename=None,
    cmap="BrBG_r",
    title="Great Lakes 12-Month NBS Climatology Exceedance Outlook"
):
    """
    Plot a 12-panel spatial map of Climatology Exceedance Probability (CEP).

    Renders one map per forecast month (up to twelve) over the Great Lakes
    basin, shading each lake polygon by its CEP value on a diverging
    wetter-to-drier colour scale.

    Parameters
    ----------
    df_cep : pd.DataFrame
        CEP data. Must contain 'forecast_month', 'lake', and the column named
        by ``value_col`` (and a 'cep' column used for labels).
    model : str, optional
        If given, plot only this model. If None, the multi-model mean across
        models is plotted.
    value_col : str, default "cep"
        Column used to colour each lake polygon.
    filename : str, optional
        Path to save the figure. If None, the figure is displayed only.
    cmap : str, default "BrBG_r"
        Matplotlib colormap name for the diverging wetter/drier scale.
    title : str, optional
        Base figure title; the selected model (or "Multi-model Mean") is
        appended.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    df = df_cep.copy()
    df["date"] = pd.to_datetime(df["forecast_month"], format="%Y-%m")
    df["lake"] = df["lake"].str.lower().str.strip()

    if model is not None:
        df = df[df["model"] == model].copy()
        plot_title = f"{title} - {model}"
    else:
        df = (
            df
            .groupby(["lake", "forecast_month", "date"], as_index=False)
            .mean(numeric_only=True)
        )
        plot_title = f"{title} - Multi-model Mean"

    # Natural Earth lake polygon file
    lake_shp = shpreader.natural_earth(
        resolution="10m",
        category="physical",
        name="lakes"
    )

    lake_records = list(shpreader.Reader(lake_shp).records())

    # Match Natural Earth names to your lake names
    lake_name_map = {
        "Lake Superior": "superior",
        "Lake Michigan": "michigan-huron",
        "Lake Huron": "michigan-huron",
        "Lake Erie": "erie",
        "Lake Ontario": "ontario",
    }

    great_lake_records = []
    for rec in lake_records:
        name = rec.attributes.get("name")
        if name in lake_name_map:
            great_lake_records.append((lake_name_map[name], rec.geometry))

    months = (
        df[["forecast_month", "date"]]
        .drop_duplicates()
        .sort_values("date")
        .head(12)
    )

    norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    cbar_label = "Wetter     ←     Normal     →     Drier"

    cmap_obj = plt.get_cmap(cmap)

    # 4 rows x 3 columns = 3 across, 4 down
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(11, 11),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True
    )

    axes = axes.flatten()

    for i, (_, month_row) in enumerate(months.iterrows()):
        ax = axes[i]

        forecast_month = month_row["forecast_month"]
        date = month_row["date"]
        month_df = df[df["forecast_month"] == forecast_month]

        ax.set_extent([-93, -74, 41, 50], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.35)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)

        states = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none"
        )
        ax.add_feature(states, edgecolor="gray", linewidth=0.4)

        for lake, geom in great_lake_records:
            row = month_df[month_df["lake"] == lake]

            if row.empty:
                facecolor = "lightgray"
                label = "NA"
            else:
                value = row[value_col].iloc[0]
                cep = row["cep"].iloc[0]

                if pd.isna(value):
                    facecolor = "lightgray"
                    label = "NA"
                else:
                    facecolor = cmap_obj(norm(value))
                    label = f"{cep:.2f}"

            ax.add_geometries(
                [geom],
                crs=ccrs.PlateCarree(),
                facecolor=facecolor,
                edgecolor="black",
                linewidth=0.8,
                zorder=4
            )

            # Label each lake with CEP value
            #point = geom.representative_point()
            #ax.text(
            #    point.x,
            #    point.y,
            #    label,
            #    ha="center",
            #    va="center",
            #    fontsize=8,
            #    fontweight="bold",
            #    transform=ccrs.PlateCarree(),
            #    zorder=5
            #)

        ax.set_aspect(1.25)
        ax.set_axis_off()

        ax.text(
            0.97,
            0.96,
            date.strftime("%b %Y"),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=13,
            color="black",
            zorder=2000
)

        # Draw black box around each subplot
        rect = mpatches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            zorder=1000,
            clip_on=False
        )

        ax.add_patch(rect)

    for j in range(len(months), 12):
        axes[j].set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm._A = []

    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.04
    )

    cbar.set_label(cbar_label)

    ticks = [0, 0.25, 0.5, 0.75, 1]

    # Bottom labels
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])

    fig.suptitle(plot_title, fontsize=16, fontweight="bold")

    if filename:
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

    return fig