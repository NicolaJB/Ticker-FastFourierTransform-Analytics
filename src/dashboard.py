# src/dashboard.py
import os
import pandas as pd
import numpy as np


def generate_financial_dashboard(
    fft_features_with_anomalies: pd.DataFrame,
    charts_dir: str = "charts",
    anomaly_threshold_pct: float = 5.0
) -> pd.DataFrame:
    """
    Build an instrument-level risk summary table.

    Outputs:
    - charts/financial_health_summary.csv

    Returns:
    - summary_df sorted by % anomalous windows (descending)
    """

    os.makedirs(charts_dir, exist_ok=True)

    summaries = []

    for inst, inst_df in fft_features_with_anomalies.groupby("Instrument Code"):
        total_windows = len(inst_df)
        num_anomalies = (inst_df["anomaly"] == -1).sum()

        pct_anomalies = (
            (num_anomalies / total_windows) * 100
            if total_windows > 0
            else 0.0
        )

        summaries.append({
            "Instrument": inst,
            "Total Windows": total_windows,
            "Anomalies": num_anomalies,
            "% Anomalous": round(pct_anomalies, 2),
            "Avg Total Power": inst_df["total_power"].mean(),
            "Dominant Frequency": (
                inst_df["dominant_frequency"].mode().iloc[0]
                if not inst_df["dominant_frequency"].dropna().empty
                else np.nan
            ),
            "High Risk": pct_anomalies >= anomaly_threshold_pct
        })

    summary_df = (
        pd.DataFrame(summaries)
        .sort_values("% Anomalous", ascending=False)
        .reset_index(drop=True)
    )

    output_path = os.path.join(charts_dir, "financial_health_summary.csv")
    summary_df.to_csv(output_path, index=False)

    print(f"Financial dashboard summary saved to {output_path}")

    return summary_df
