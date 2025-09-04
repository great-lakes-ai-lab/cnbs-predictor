# forecast_visualizer.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_forecast(df, filename="forecast/CNBS_forecasts.png"):
    """
    Static Matplotlib plot for Great Lakes CNBS forecasts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Pivoted dataframe with columns:
        ["cfs_run", "forecast_month", "model", "lake", "precipitation", "evaporation", "runoff", "nbs"]
    filename : str, optional
        Path to save the PNG. Defaults to 'forecast/CNBS_forecasts.png'.
    """
    # Ensure output folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

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
    plt.savefig(filename)
    plt.show()


def plot_forecast_interactive(df, filename="forecast/CNBS_forecasts_interactive.html"):
    """
    Interactive Plotly plot for Great Lakes CNBS forecasts.

    Parameters
    ----------
    df : pd.DataFrame
        Pivoted dataframe with columns:
        ["cfs_run", "forecast_month", "model", "lake", "precipitation", "evaporation", "runoff", "nbs"]
    filename : str, optional
        File path for saving the interactive HTML. Defaults to 'forecast/CNBS_forecasts_interactive.html'.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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

    # Save HTML
    fig.write_html(filename)

    # Show plot
    fig.show()