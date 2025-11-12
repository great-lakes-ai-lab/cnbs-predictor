# forecast_visualizer.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore", message="This axis already has a converter set")

def plot_cnbs_forecast(df, filename=None):
    """
    Static Matplotlib plot for Great Lakes CNBS forecasts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Pivoted dataframe with columns:
        ["cfs_run", "forecast_month", "model", "lake", "precipitation", "evaporation", "runoff", "nbs"]
    filename : str, optional
        Path to save the PNG. If None, the plot is displayed only and not saved.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["forecast_month"], format="%Y-%m")

    # Unique models
    models_info = df["model"].unique().tolist()

    # Colors
    custom_colors = ['blue', 'green', 'red', 'purple', 'orange',
                     'brown', 'pink', 'olive', 'cyan', 'magenta']
    if len(models_info) > len(custom_colors):
        raise ValueError(
            f"Number of models ({len(models_info)}) exceeds available colors "
            f"({len(custom_colors)}). Add more colors."
        )
    model_colors = {model: custom_colors[i] for i, model in enumerate(models_info)}

    # Lakes and components
    lakes = ["superior", "michigan-huron", "erie", "ontario"]
    components = ["precipitation", "evaporation", "runoff", "nbs"]

    # Create 4x4 subplot grid
    fig, axs = plt.subplots(4, 4, figsize=(18, 8))

    shared_y_min, shared_y_max = float('inf'), float('-inf')
    col4_y_min, col4_y_max = float('inf'), float('-inf')

    for row, lake in enumerate(lakes):
        for col, comp in enumerate(components):
            ax = axs[row, col]
            lake_df = df[df["lake"] == lake]

            # Confidence intervals
            lower = lake_df.groupby("forecast_month")[comp].quantile(0.025).reset_index()
            upper = lake_df.groupby("forecast_month")[comp].quantile(0.975).reset_index()
            lower["date"] = pd.to_datetime(lower["forecast_month"], format="%Y-%m")
            upper["date"] = pd.to_datetime(upper["forecast_month"], format="%Y-%m")

            # Plot each model
            for model_name in models_info:
                model_mean = lake_df[lake_df["model"] == model_name]
                mean_df = model_mean.groupby("forecast_month").mean(numeric_only=True).reset_index()
                mean_df["date"] = pd.to_datetime(mean_df["forecast_month"], format="%Y-%m")

                ax.plot(mean_df["date"], mean_df[comp], marker='o', markersize=3,
                        color=model_colors[model_name], label=model_name)

            ax.fill_between(lower["date"], lower[comp], upper[comp],
                            color='gray', alpha=0.2, label='95%')

            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_yticks(np.arange(-1000, 1000, 50))

            # Track y-limits
            current_y_min = lower[comp].min()
            current_y_max = upper[comp].max()
            if col < 3:
                shared_y_min = min(shared_y_min, current_y_min)
                shared_y_max = max(shared_y_max, current_y_max)
            else:
                col4_y_min = min(col4_y_min, current_y_min)
                col4_y_max = max(col4_y_max, current_y_max)

            if row == 0:
                ax.set_title(["Precipitation", "Evaporation", "Runoff", "NBS"][col], fontsize=14)

            if row == 3:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
                ax.tick_params(axis='x', rotation=45)
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(["Superior", "Mich-Huron", "Erie", "Ontario"][row], fontsize=14)
            elif col == 3:
                ax.set_ylabel("")
            else:
                ax.set_yticklabels([])

    # Apply y-axis limits
    for row_idx in range(4):
        for col_idx in range(4):
            if col_idx < 3:
                axs[row_idx, col_idx].set_ylim(shared_y_min-10, shared_y_max+10)
            else:
                axs[row_idx, col_idx].set_ylim(col4_y_min-10, col4_y_max+10)
            axs[row_idx, col_idx].set_xlim(df["date"].min(), df["date"].max())

    fig.suptitle("Great Lakes 12-Month CNBS [mm] Forecast", fontsize=16)
    axs[0, 3].legend(loc='lower left', bbox_to_anchor=(1, 0.04), fontsize=12)

    plt.tight_layout()

    # Save only if filename is provided
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')

    plt.show()


def plot_cnbs_forecast_interactive(df, filename=None):
    """
    Interactive Plotly plot for Great Lakes CNBS forecasts.

    Parameters
    ----------
    df : pd.DataFrame
        Pivoted dataframe with columns:
        ["cfs_run", "forecast_month", "model", "lake", "precipitation", "evaporation", "runoff", "nbs"]
    filename : str, optional
        File path for saving the interactive HTML. Defaults to None.
    """
   
    df = df.copy()
    df["date"] = pd.to_datetime(df["forecast_month"], format="%Y-%m")

    models_info = df["model"].unique().tolist()
    custom_colors = ['blue', 'green', 'red', 'purple', 'orange',
                     'brown', 'pink', 'olive', 'cyan', 'magenta']
    if len(models_info) > len(custom_colors):
        raise ValueError(f"Number of models ({len(models_info)}) exceeds available colors ({len(custom_colors)}).")

    model_colors = {model: custom_colors[i] for i, model in enumerate(models_info)}
    lakes = ["superior", "michigan-huron", "erie", "ontario"]
    components = ["precipitation", "evaporation", "runoff", "nbs"]

    # Create 4x4 subplot grid
    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=['Precipitation', 'Evaporation', 'Runoff', 'NBS'],
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.05, horizontal_spacing=0.008
    )

    global_y_min, global_y_max = float('inf'), float('-inf')

    for row, lake in enumerate(lakes):
        for col, comp in enumerate(components):
            lake_df = df[df["lake"] == lake]

            # 95% confidence bounds across all models
            lower = lake_df.groupby("forecast_month")[comp].quantile(0.025).reset_index()
            upper = lake_df.groupby("forecast_month")[comp].quantile(0.975).reset_index()

            for model_name in models_info:
                model_mean = lake_df[lake_df['model'] == model_name]
                mean_df = model_mean.groupby("forecast_month").mean(numeric_only=True).reset_index()
                mean_df['date'] = pd.to_datetime(mean_df["forecast_month"], format="%Y-%m")

                # Shaded CI
                fig.add_trace(go.Scatter(
                    x=pd.concat([mean_df['date'], mean_df['date'][::-1]]),
                    y=pd.concat([lower[comp], upper[comp][::-1]]),
                    fill='toself', fillcolor='rgba(169, 169, 169, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'), showlegend=False,
                    hovertemplate='Date: %{x}<br>Conf: 95%<br>Value: %{y}'
                ), row=row+1, col=col+1)

                # Mean line per model
                fig.add_trace(go.Scatter(
                    x=mean_df['date'], y=mean_df[comp],
                    mode='lines+markers', name=model_name,
                    line=dict(shape='linear', width=2, color=model_colors[model_name]),
                    marker=dict(size=4),
                    showlegend=(row == 0 and col == 0),
                    hovertemplate='Date: %{x}<br>Value: %{y}<br>Model: ' + model_name
                ), row=row+1, col=col+1)

                # Update global y-axis
                global_y_min = min(global_y_min, lower[comp].min())
                global_y_max = max(global_y_max, upper[comp].max())

            # Dashed y=0 line
            fig.add_trace(go.Scatter(
                x=[mean_df['date'].min(), mean_df['date'].max()],
                y=[0, 0], mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ), row=row+1, col=col+1)

            # Solid box around subplot
            fig.update_xaxes(
                tickformat='%m-%Y',
                tickangle=-45,
                tickvals=mean_df["date"],
                showticklabels=(row == 3),
                showline=True,
                linewidth=2,
                linecolor='black',
                row=row+1, col=col+1
            )
            fig.update_yaxes(
                title_text=lake.title() if col == 0 else None,
                showline=True,
                linewidth=2,
                linecolor='black',
                row=row+1, col=col+1
            )

    # Consistent y-axis across all subplots
    for r in range(1, 5):
        for c in range(1, 5):
            fig.update_yaxes(range=[global_y_min - 10, global_y_max + 10], row=r, col=c)

    # Gridlines
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')

    # Layout
    fig.update_layout(
        title_text="Great Lakes 12-Month CNBS [mm] Forecast",
        title_x=0.5,
        height=800, width=1500,
        showlegend=True,
        legend=dict(x=1.01, y=1.01, traceorder='normal', orientation='v', title='Model'),
        plot_bgcolor='white'
    )

    # Save only if filename is provided
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.write_html(filename)

    # Show plot
    fig.show()

def plot_nbs_forecast(df_filtered, prob, filename=None):
    """
    Plot NBS forecasts for the Great Lakes with ensemble model lines, 95% range, and climatology.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Forecast data containing 'forecast_month', 'lake', 'model', and 'nbs'.
    prob : pd.DataFrame
        Probability of exceedance data containing 'month', 'lake', 'prob_exceedance', and 'value'.
    """
    
    # Make a copy to avoid modifying the original
    df = df_filtered.copy()
    df["date"] = pd.to_datetime(df["forecast_month"], format="%Y-%m")

    # Unique models
    models_info = df["model"].unique().tolist()

    # Colors
    custom_colors = ['blue', 'green', 'red', 'purple', 'orange',
                     'brown', 'pink', 'olive', 'cyan', 'magenta']
    if len(models_info) > len(custom_colors):
        raise ValueError(
            f"Number of models ({len(models_info)}) exceeds available colors "
            f"({len(custom_colors)}). Add more colors."
        )
    model_colors = {model: custom_colors[i] for i, model in enumerate(models_info)}

    # Lakes and components
    lakes = ["superior", "michigan-huron", "erie", "ontario"]
    components = ["nbs"]

    # Begin plotting
    fig, axs = plt.subplots(len(lakes), len(components), figsize=(10, 12), squeeze=False)
    fig.suptitle("Great Lakes 12-Month NBS [mm] Forecast", fontsize=16, x=0.2, ha='left')

    shared_y_min, shared_y_max = float('inf'), float('-inf')
    col4_y_min, col4_y_max = float('inf'), float('-inf')

    for row, lake in enumerate(lakes):
        for col, comp in enumerate(components):
            ax = axs[row, col]
            lake_df = df[df["lake"] == lake]

            # Confidence intervals
            lower = lake_df.groupby("forecast_month")[comp].quantile(0.025).reset_index()
            upper = lake_df.groupby("forecast_month")[comp].quantile(0.975).reset_index()
            lower["date"] = pd.to_datetime(lower["forecast_month"], format="%Y-%m")
            upper["date"] = pd.to_datetime(upper["forecast_month"], format="%Y-%m")

            # Plot each model
            for model_name in models_info:
                model_mean = lake_df[lake_df["model"] == model_name]
                mean_df = model_mean.groupby("forecast_month").mean(numeric_only=True).reset_index()
                mean_df["date"] = pd.to_datetime(mean_df["forecast_month"], format="%Y-%m")

                ax.plot(mean_df["date"], mean_df[comp], marker='o', markersize=3,
                        color=model_colors[model_name], label=model_name)

            # Fill 95% range
            ax.fill_between(lower["date"], lower[comp], upper[comp],
                            color='gray', alpha=0.2, label='95%')

            # Climatology (PoE = 0.5)
            lake_prob = prob.query(f"lake == '{lake}' and prob_exceedance == 0.5")
            lake_prob.plot(x="date", y="value", ax=ax, label="Climatology",
                           color="black", linestyle="--", linewidth=2)

            # Plot formatting
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_yticks(np.arange(-1000, 1000, 100))
            ax.set_xlabel("")

            # Track y-limits
            current_y_min = lower[comp].min()
            current_y_max = upper[comp].max()
            if col < 3:
                shared_y_min = min(shared_y_min, current_y_min)
                shared_y_max = max(shared_y_max, current_y_max)
            else:
                col4_y_min = min(col4_y_min, current_y_min)
                col4_y_max = max(col4_y_max, current_y_max)

            # X-axis formatting
            if row == 3:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
                ax.tick_params(axis='x', rotation=45)
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')
            else:
                ax.set_xticklabels([])

            # Y-axis labels
            if col == 0:
                ax.set_ylabel(["Superior", "Mich-Huron", "Erie", "Ontario"][row], fontsize=14)
            elif col == 3:
                ax.set_ylabel("")
            else:
                ax.set_yticklabels([])

    # Apply y-axis limits
    for row_idx in range(len(lakes)):
        for col_idx in range(len(components)):
            if col_idx < 3:
                axs[row_idx, col_idx].set_ylim(shared_y_min-10, shared_y_max+10)
            else:
                axs[row_idx, col_idx].set_ylim(col4_y_min-10, col4_y_max+10)
            axs[row_idx, col_idx].set_xlim(df["date"].min(), df["date"].max())

    # Legend only on top-left subplot
    axs[0, 0].legend(loc='lower left', bbox_to_anchor=(1, 0.3), fontsize=12)
    for r in range(len(lakes)):
        for c in range(len(components)):
            if not (r == 0 and c == 0):
                if axs[r, c].get_legend() is not None:
                    axs[r, c].legend_.remove()

    plt.tight_layout()

        # Save only if filename is provided
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')

    # Show plot
    plt.show()

def plot_prob_exceed(df_filtered, prob, filename=None):
    """
    Plot Great Lakes 12-Month NBS probability of exceedance for multiple models.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Forecast data containing 'forecast_month', 'lake', 'model', and 'nbs'.
    prob : pd.DataFrame
        Probability of exceedance data with 'month', 'lake', 'prob_exceedance', and 'value'.
    filename : str, optional
        Path to save the figure. If None, the plot is displayed only.
    """

    # --- Copy and prepare dataframe ---
    df = df_filtered.copy()
    df["date"] = pd.to_datetime(df["forecast_month"], format="%Y-%m")

    # --- Lakes ---
    lakes = ["superior", "michigan-huron", "erie", "ontario"]

    # --- Colors for models ---
    models_info = df["model"].unique().tolist()
    custom_colors = ['blue', 'green', 'red', 'purple', 'orange',
                     'brown', 'pink', 'olive', 'cyan', 'magenta']
    if len(models_info) > len(custom_colors):
        raise ValueError(
            f"Number of models ({len(models_info)}) exceeds available colors "
            f"({len(custom_colors)}). Add more colors."
        )
    model_colors = {model: custom_colors[i] for i, model in enumerate(models_info)}

    # --- Helper: find PoE for given value ---
    def get_poe_from_df(value, month, lake, poe_df):
        """Interpolate the probability of exceedance for a given lake/month/value."""
        subset = poe_df[(poe_df["lake"] == lake) & (poe_df["month"] == month)]
        if subset.empty:
            return np.nan
        x = subset["value"].values
        y = subset["prob_exceedance"].values

        # Sort to ensure monotonic interpolation
        sort_idx = np.argsort(x)
        x, y = x[sort_idx], y[sort_idx]

        # Boundaries
        if value <= x.min():
            return y.max()
        if value >= x.max():
            return y.min()
        return np.interp(value, x, y[::-1] if x[0] > x[-1] else y)

    # --- Month map ---
    MONTH_MAP = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                 5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    # --- Initialize figure ---
    fig, axs = plt.subplots(len(lakes), 1, figsize=(10, 12), squeeze=False)
    fig.suptitle("Great Lakes 12-Month Probability of Exceedance (NBS)", fontsize=16, x=0.17, ha='left')

    # --- Loop over lakes ---
    for row, lake in enumerate(lakes):
        ax = axs[row, 0]
        lake_df = df[df["lake"] == lake]

        # Plot each model
        for model_name in models_info:
            model_mean = lake_df[lake_df["model"] == model_name]
            mean_df = model_mean.groupby("forecast_month").mean(numeric_only=True).reset_index()
            mean_df["date"] = pd.to_datetime(mean_df["forecast_month"], format="%Y-%m")

            poes = []
            for _, row_data in mean_df.iterrows():
                month_num = row_data["date"].month
                month_name = MONTH_MAP[month_num]
                poe = get_poe_from_df(row_data["nbs"], month_name, lake, prob)
                poes.append(poe)
            mean_df["PoE"] = poes

            ax.plot(mean_df["date"], mean_df["PoE"], marker='o', markersize=3,
                    color=model_colors[model_name], label=model_name)

        # Formatting
        ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_xlim(df["date"].min(), df["date"].max())
        ax.set_ylabel(["Superior", "Mich-Huron", "Erie", "Ontario"][row], fontsize=14)

        # Format x-axis
        if row == len(lakes) - 1:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
            ax.tick_params(axis='x', rotation=45)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        else:
            ax.set_xticklabels([])

    # --- Legend only on top plot ---
    axs[0, 0].legend(loc='lower left', bbox_to_anchor=(1, 0.525), fontsize=12)

    plt.tight_layout()

    # Save the figure if filename is provided
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')

    # Show the plot
    plt.show()


