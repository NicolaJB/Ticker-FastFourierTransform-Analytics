from src.InstrumentDataProcessor import InstrumentDataProcessor
from src.FFTFeatureExtractor import FFTFeatureExtractor
from src.AnomalyDetector import AnomalyDetector
import pandas as pd
import numpy as np

# Step 1: Process your ticker data
processor = InstrumentDataProcessor(
    "data/data.txt",
    "data/StaticFields.txt",
    "data/DynamicFields.txt"
)
output_file = processor.process()

# Step 2: Load CSV
df = pd.read_csv(output_file)
print(f"Loaded {len(df)} rows from {output_file}")

# Step 2b: Keep only numeric values for FFT
# Ensure 'Value' column is numeric (coerce non-numeric to NaN)
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df = df.dropna(subset=['Value'])
print(f"Filtered to {len(df)} numeric rows for FFT")

# Step 3: Extract FFT features per rolling window
fft_extractor = FFTFeatureExtractor(sampling_rate=1, window_size=20, step_size=5)
fft_features_df = fft_extractor.compute_rolling_features(
    df,
    value_col="Value",
    instrument_col="Instrument Code",
    timestamp_col="Timestamp"
)

# Step 4: Save FFT features
fft_features_df.to_csv("fft_features.csv", index=False)
print("FFT features saved to fft_features.csv")
print(fft_features_df.head())

# Step 5 (Optional): Run Anomaly Detection
feature_cols = ["dominant_frequency", "total_power", "spectral_entropy"]
anomaly_detector = AnomalyDetector(contamination=0.05)
fft_features_with_anomalies = anomaly_detector.fit_predict(fft_features_df, feature_cols)
fft_features_with_anomalies.to_csv("fft_features_with_anomalies.csv", index=False)
print("Anomaly detection completed and saved to fft_features_with_anomalies.csv")
print(fft_features_with_anomalies.head())
