import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

class FFTFeatureExtractor:
    def __init__(self, sampling_rate=1, window_size=20, step_size=5):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.step_size = step_size

    def compute_fft_features(self, series):
        series = pd.to_numeric(series, errors="coerce").dropna().values.astype(float)
        n = len(series)

        if n == 0:
            return None

        fft_values = fft(series)
        freqs = fftfreq(n, d=1 / self.sampling_rate)
        power = np.abs(fft_values) ** 2 / n

        # Exclude DC component (freq=0)
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power = power[pos_mask]

        # Normalize power to sum to 1
        total_power = np.sum(power)
        if total_power > 0:
            power = power / total_power
        else:
            power = np.zeros_like(power)

        dominant_freq = freqs[np.argmax(power)] if len(power) > 0 else 0.0
        spectral_entropy = -np.sum(power * np.log2(power + 1e-12))

        return {
            "dominant_frequency": dominant_freq,
            "total_power": total_power,
            "spectral_entropy": spectral_entropy,
        }

    def compute_rolling_features(
        self,
        df,
        value_col="Value",
        instrument_col="Instrument Code",
        timestamp_col="Timestamp",
    ):
        results = []

        for inst in df[instrument_col].unique():
            inst_df = df[df[instrument_col] == inst].sort_values(timestamp_col)

            series = inst_df[value_col]
            timestamps = inst_df[timestamp_col].values

            for start in range(0, len(series) - self.window_size + 1, self.step_size):
                end = start + self.window_size
                window_series = series.iloc[start:end]

                fft_features = self.compute_fft_features(window_series)
                if fft_features is None:
                    continue

                fft_features.update({
                    instrument_col: inst,
                    "window_start": timestamps[start],
                    "window_end": timestamps[end - 1],
                    "rolling_mean": window_series.mean(),
                    "rolling_std": window_series.std(),
                    "rolling_skew": window_series.skew(),
                })

                results.append(fft_features)

        return pd.DataFrame(results)
