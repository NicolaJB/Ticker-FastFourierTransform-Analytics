import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

class FFTFeatureExtractor:
    def __init__(self, sampling_rate=1, window_size=20, step_size=5):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.step_size = step_size

    def compute_fft_features(self, series):
        # Ensure numeric only, skip non-numeric
        series = pd.to_numeric(series, errors='coerce')
        series = series.dropna().values.astype(float)
        n = len(series)
        if n == 0:
            return None

        fft_values = fft(series)
        freqs = fftfreq(n, d=1/self.sampling_rate)
        power = np.abs(fft_values)**2 / n

        # Positive frequencies only
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        power = power[pos_mask]

        dominant_freq = freqs[np.argmax(power)]
        total_power = np.sum(power)
        spectral_entropy = -np.sum((power/total_power) * np.log2(power/total_power + 1e-12))

        return {
            "dominant_frequency": dominant_freq,
            "total_power": total_power,
            "spectral_entropy": spectral_entropy
        }

    def compute_rolling_features(self, df, value_col="Value", instrument_col="Instrument Code", timestamp_col="Timestamp"):
        result_list = []

        instruments = df[instrument_col].unique()
        for inst in instruments:
            inst_df = df[df[instrument_col] == inst].sort_values(timestamp_col)
            series = inst_df[value_col]
            timestamps = inst_df[timestamp_col].values

            for start_idx in range(0, len(series) - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                window_series = series[start_idx:end_idx]
                window_start = timestamps[start_idx]
                window_end = timestamps[end_idx - 1]

                features = self.compute_fft_features(window_series)
                if features:  # skip windows with no numeric data
                    features[instrument_col] = inst
                    features["window_start"] = window_start
                    features["window_end"] = window_end
                    result_list.append(features)

        return pd.DataFrame(result_list)
