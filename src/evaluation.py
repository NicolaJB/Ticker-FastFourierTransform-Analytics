import numpy as np
import pandas as pd


def compare_feature_sets(df, fft_cols, baseline_cols):
    """
    Compare FFT feature set vs baseline (time-domain) features.

    Returns:
    - Average variance per feature set
    - Variance ratio (FFT / baseline)
    - Normalized signal improvement (%)
    - Optional: anomaly spread per feature set (future expansion)
    """
    # Ensure numeric only
    fft_df = df[fft_cols].select_dtypes(include=np.number)
    baseline_df = df[baseline_cols].select_dtypes(include=np.number)

    # Compute mean variance per feature set
    fft_variance = fft_df.var().mean()
    baseline_variance = baseline_df.var().mean()

    # Compute ratio
    variance_ratio = fft_variance / baseline_variance if baseline_variance != 0 else None

    # Compute normalized improvement (%) over baseline
    signal_improvement_pct = ((fft_variance - baseline_variance) / baseline_variance * 100
                              if baseline_variance != 0 else None)

    return {
        "fft_avg_variance": fft_variance,
        "baseline_avg_variance": baseline_variance,
        "variance_ratio": variance_ratio,
        "signal_improvement_pct": signal_improvement_pct
    }
