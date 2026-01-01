# src/visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def ensure_dir(directory: str):
    os.makedirs(directory, exist_ok=True)


def parse_window_end_to_minutes(t):
    """
    Convert window_end into minutes since start.
    Accepts:
    - HH:MM:SS:MS
    - MM:SS:MS
    - seconds (int/float)
    Returns float minutes or NaN.
    """
    try:
        if pd.isna(t):
            return np.nan

        if isinstance(t, (int, float)):
            return float(t) / 60.0

        parts = [float(p) for p in str(t).split(":")]

        if len(parts) == 4:      # HH:MM:SS:MS
            h, m, s, ms = parts
            total_seconds = h * 3600 + m * 60 + s + ms / 1000
        elif len(parts) == 3:    # MM:SS:MS
            m, s, ms = parts
            total_seconds = m * 60 + s + ms / 1000
        elif len(parts) == 2:    # SS:MS
            s, ms = parts
            total_seconds = s + ms / 1000
        else:
            return np.nan

        return total_seconds / 60.0

    except Exception:
        return np.nan


# -------------------------------------------------
# Figure 1: Dominant Frequency Histogram
# -------------------------------------------------
def plot_dominant_frequency_histogram(df, output_dir="charts", bins=50, fig_num=1):
    ensure_dir(output_dir)
    values = df["dominant_frequency"].dropna()
    if values.empty:
        print("No dominant_frequency data to plot.")
        return

    plt.figure(fig_num, figsize=(8, 5))
    plt.hist(values, bins=bins, color="steelblue", edgecolor="black")
    plt.title("Figure 1: Distribution of Dominant Frequencies")
    plt.xlabel("Dominant Frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"figure_{fig_num}_dominant_frequency.png"))
    plt.close()


# -------------------------------------------------
# Figures 3–7: Total Power Over Time per Instrument
# -------------------------------------------------
def plot_anomalies_over_time(
    df,
    instrument_code,
    value_col="total_power",
    output_dir="charts",
    n_ticks=10,
    fig_num=3,
    highlight_high_risk=True
):
    ensure_dir(output_dir)
    inst_df = df[df["Instrument Code"] == instrument_code].copy()
    if inst_df.empty:
        print(f"[Figure {fig_num}] No rows for {instrument_code}")
        return

    # ---- unified time axis (minutes since start)
    inst_df["t_min"] = inst_df["window_end"].apply(parse_window_end_to_minutes)
    inst_df = inst_df.dropna(subset=["t_min", value_col])
    inst_df = inst_df.sort_values("t_min")
    if inst_df.empty:
        print(f"[Figure {fig_num}] No valid parsed data for {instrument_code}")
        return

    # ---- plotting with 1D arrays
    plt.figure(fig_num, figsize=(12, 5))
    plt.plot(
        np.array(inst_df["t_min"].tolist()),
        np.array(inst_df[value_col].tolist()),
        label="Total Power",
        linewidth=2,
        marker="o",
        color="royalblue"
    )

    # anomalies
    anomalies = inst_df[inst_df["anomaly"] == -1]
    if not anomalies.empty:
        plt.scatter(
            np.array(anomalies["t_min"].tolist()),
            np.array(anomalies[value_col].tolist()),
            color="crimson",
            s=60,
            zorder=5,
            label="Anomalies"
        )

    plt.title(f"Figure {fig_num}: Total Power Over Time ({instrument_code})")
    plt.xlabel("Time (minutes since start)")
    plt.ylabel("Total Power")
    plt.legend()
    plt.grid(alpha=0.3)

    # reduce tick clutter
    if len(inst_df) > n_ticks:
        idx = np.linspace(0, len(inst_df) - 1, n_ticks).astype(int)
        plt.xticks(inst_df["t_min"].iloc[idx])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"figure_{fig_num}_total_power_{instrument_code}.png"))
    plt.close()


# -------------------------------------------------
# Figure 2: Top Anomalies Bar Chart
# -------------------------------------------------
def plot_top_anomalies_bar(summary_df, top_n=20, output_dir="charts", fig_num=2):
    ensure_dir(output_dir)
    df = summary_df.head(top_n).copy()
    if df.empty:
        print("No dashboard data to plot.")
        return

    colors = ["red" if hr else "orange" for hr in df["High Risk"]]

    plt.figure(fig_num, figsize=(12, 5))
    bars = plt.bar(df["Instrument"], df["% Anomalous"], color=colors)
    for bar, hr in zip(bars, df["High Risk"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold" if hr else "normal",
            color="darkred" if hr else "black"
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("% Anomalous Windows")
    plt.title("Figure 2: Top Instruments by % Anomalous Windows")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"figure_{fig_num}_top_anomalies.png"))
    plt.close()
