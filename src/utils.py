import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_ticker_data(
        instruments=None,
        start_date="2025-01-01",
        num_days=100,
        freq="D",
        seed=42
):
    """
    Generate synthetic multi-instrument ticker data for FFT + anomaly detection demos.

    :param instruments: list of ticker symbols
    :param start_date: string, start date
    :param num_days: number of days to simulate
    :param freq: frequency string, e.g., 'D' for daily
    :param seed: random seed for reproducibility
    :return: Pandas DataFrame with columns ['Date', 'Instrument', 'Value']
    """
    np.random.seed(seed)
    if instruments is None:
        instruments = ["AAPL", "SPY", "GOOG", "MSFT"]

    dates = pd.date_range(start=start_date, periods=num_days, freq=freq)
    data = []

    for inst in instruments:
        # Generate a synthetic price series with trend + seasonal + noise
        trend = np.linspace(100, 120, num_days)  # linear trend
        seasonal = 5 * np.sin(np.linspace(0, 4 * np.pi, num_days))  # sine wave
        noise = np.random.normal(0, 1, num_days)  # Gaussian noise
        prices = trend + seasonal + noise

        # Convert prices to "returns" if desired
        returns = np.diff(np.log(prices + 1e-9), prepend=np.log(prices[0] + 1e-9))

        for d, r in zip(dates, returns):
            data.append([d, inst, r])

    df = pd.DataFrame(data, columns=["Date", "Instrument Code", "Value"])
    return df
