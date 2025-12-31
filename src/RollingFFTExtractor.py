import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

class RollingFFTExtractor:
    def __init__(self, window_size=20, sampling_rate=1):
        """
        :param window_size: Number of observations per rolling window
        :param sampling_rate: Observations per unit time
        """
        self.window_size = window_size
        self.sampling_rate = sampling_rate

    def compute_fft_features(self, series):
        series = np.array(series, dtype=float)
        n = len(series)
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
        """
        Compute FFT features for each instrument using rolling windows.
        Returns a DataFrame with one row per window.
        """
        results = []
        instruments = df[instrument_col].unique()

        for inst in instruments:
            inst_df = df[df[instrument_col] == inst].sort_values(timestamp_col)
            series = inst_df[value_col].values

            for start in range(len(series) - self.window_size + 1):
                window_series = series[start:start+self.window_size]
                window_start_time = inst_df[timestamp_col].iloc[start]
                window_end_time = inst_df[timestamp_col].iloc[start+self.window_size-1]

                fft_features = self.compute_fft_features(window_series)
                fft_features.update({
                    instrument_col: inst,
                    "window_start": window_start_time,
                    "window_end": window_end_time
                })
                results.append(fft_features)

        return pd.DataFrame(results)
