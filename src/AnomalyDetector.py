import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, contamination=0.05, random_state=42):
        """
        :param contamination: Fraction of outliers expected in the dataset
        """
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit_predict(self, feature_df, feature_cols=None):
        """
        Fit Isolation Forest and predict anomalies.
        :param feature_df: DataFrame of FFT features per rolling window
        :param feature_cols: Columns to use for anomaly detection (optional)
        :return: DataFrame with anomaly flag (1 = normal, -1 = anomaly)
        """
        if feature_cols is None:
            # Default: all numeric columns except instrument/time identifiers
            feature_cols = feature_df.select_dtypes(include=np.number).columns.tolist()

        X = feature_df[feature_cols].values
        feature_df['anomaly'] = self.model.fit_predict(X)
        return feature_df

    def get_anomalies(self, feature_df):
        """
        Return only rows flagged as anomalies
        """
        return feature_df[feature_df['anomaly'] == -1]

    def detect_per_instrument(self, feature_df, instrument_col="Instrument Code", feature_cols=None):
        """
        Run anomaly detection separately for each instrument
        :param feature_df: DataFrame with FFT features
        :param instrument_col: Column containing instrument identifiers
        :param feature_cols: Columns to use for anomaly detection
        :return: DataFrame with anomaly flags per instrument
        """
        result_df = pd.DataFrame()
        instruments = feature_df[instrument_col].unique()

        for inst in instruments:
            inst_df = feature_df[feature_df[instrument_col] == inst].copy()
            inst_df = self.fit_predict(inst_df, feature_cols)
            result_df = pd.concat([result_df, inst_df], ignore_index=True)

        return result_df
