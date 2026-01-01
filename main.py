# main.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.InstrumentDataProcessor import InstrumentDataProcessor
from src.FFTFeatureExtractor import FFTFeatureExtractor
from src.AnomalyDetector import AnomalyDetector
from src.evaluation import compare_feature_sets
from src.visualization import (
    plot_dominant_frequency_histogram,
    plot_anomalies_over_time,
    plot_top_anomalies_bar
)
from src.dashboard import generate_financial_dashboard

# ------------------------------
# Create charts directory
# ------------------------------
charts_dir = "charts"
os.makedirs(charts_dir, exist_ok=True)

# ==============================
# Step 1: Process raw ticker data
# ==============================
processor = InstrumentDataProcessor(
    "data/data.txt",
    "data/StaticFields.txt",
    "data/DynamicFields.txt"
)
output_file = processor.process()  # prints output

# ==============================
# Step 2: Load processed CSV
# ==============================
df = pd.read_csv(output_file)
print(f"Loaded {len(df)} rows from {output_file}")

# Ensure numeric values for FFT
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])
print(f"Filtered to {len(df)} numeric rows for FFT")

# ==============================
# Step 3: Compute FFT + rolling features
# ==============================
fft_extractor = FFTFeatureExtractor(sampling_rate=1, window_size=20, step_size=5)
fft_features_df = fft_extractor.compute_rolling_features(
    df,
    value_col="Value",
    instrument_col="Instrument Code",
    timestamp_col="Timestamp"
)
fft_features_df.to_csv("fft_features.csv", index=False)
print("FFT features saved to fft_features.csv")

# ==============================
# Step 4: Evaluate FFT features vs baseline
# ==============================
fft_feature_cols = ["dominant_frequency", "total_power", "spectral_entropy"]
baseline_feature_cols = ["rolling_mean", "rolling_std", "rolling_skew"]

evaluation_results = compare_feature_sets(
    fft_features_df,
    fft_cols=fft_feature_cols,
    baseline_cols=baseline_feature_cols
)

print("\nFeature evaluation results:")
for k, v in evaluation_results.items():
    print(f"{k}: {v}")

fft_var = evaluation_results["fft_avg_variance"]
baseline_var = evaluation_results["baseline_avg_variance"]
signal_improvement_pct = ((fft_var - baseline_var) / baseline_var) * 100
print(f"Normalized signal improvement (FFT vs baseline): {signal_improvement_pct:.2f}%")

# ==============================
# Step 5: Run anomaly detection
# ==============================
feature_cols = [c for c in fft_features_df.select_dtypes(include=np.number).columns
                if c not in ["window_start", "window_end"]]

anomaly_detector = AnomalyDetector(contamination=0.05)
fft_features_with_anomalies = anomaly_detector.detect_per_instrument(
    fft_features_df,
    instrument_col="Instrument Code",
    feature_cols=feature_cols
)
fft_features_with_anomalies.to_csv("fft_features_with_anomalies.csv", index=False)

num_anomalies = (fft_features_with_anomalies["anomaly"] == -1).sum()
print(f"\nDetected {num_anomalies} anomalies across {fft_features_df['Instrument Code'].nunique()} instruments")

# ==============================
# Step 6: Figure 1 - Dominant frequency histogram
# ==============================
plot_dominant_frequency_histogram(
    fft_features_df,
    output_dir=charts_dir,
    fig_num=1
)

# ==============================
# Step 7: Financial dashboard + Figure 2 (Top anomalies bar chart)
# ==============================
dashboard_df = generate_financial_dashboard(
    fft_features_with_anomalies,
    charts_dir=charts_dir,
    anomaly_threshold_pct=5
)

plot_top_anomalies_bar(
    dashboard_df,
    top_n=20,
    output_dir=charts_dir,
    fig_num=2
)

print("\nTop instruments by % anomalous windows:")
print(dashboard_df.head(10))

# ==============================
# Step 8: Figures 3-7 - Total power over time for top 5 high-risk instruments
# ==============================
top_instruments = dashboard_df["Instrument"].head(5).tolist()

for idx, inst in enumerate(top_instruments, start=3):
    plot_anomalies_over_time(
        fft_features_with_anomalies,
        instrument_code=inst,
        value_col="total_power",
        output_dir=charts_dir,
        n_ticks=10,
        fig_num=idx,
        highlight_high_risk=True  # updated argument matches visualization.py
    )
