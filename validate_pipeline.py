import pandas as pd
from src.InstrumentDataProcessor import InstrumentDataProcessor
from src.FFTFeatureExtractor import FFTFeatureExtractor
from src.AnomalyDetector import AnomalyDetector

# Step 1: Process data using your existing tool
processor = InstrumentDataProcessor("data/data.txt", "data/StaticFields.txt", "data/DynamicFields.txt")
output_file = processor.process()

# Step 2: Load processed CSV
df = pd.read_csv(output_file)
print(f"Loaded {len(df)} rows from {output_file}")

# Step 3: Compute FFT features
fft_extractor = FFTFeatureExtractor(window_size=20, step_size=5)
fft_df = fft_extractor.compute_rolling_features(df, value_col="Value", instrument_col="Instrument Code", timestamp_col="Timestamp")
print(f"FFT features computed for {len(fft_df)} windows")
print(fft_df.head())

# Step 4: Run anomaly detection
feature_cols = ["dominant_frequency", "total_power", "spectral_entropy"]
anomaly_detector = AnomalyDetector(contamination=0.05)
fft_df_anomaly = anomaly_detector.fit_predict(fft_df, feature_cols=feature_cols)

print("Anomaly detection completed")
print(fft_df_anomaly.head())

# Step 5: Save results
fft_df_anomaly.to_csv("fft_features_with_anomalies_test.csv", index=False)
print("Saved FFT + anomaly results to fft_features_with_anomalies_test.csv")
