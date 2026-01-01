import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    def __init__(self, contamination=0.05, random_state=42, normalize_features=True):
        """
        :param contamination: Expected fraction of anomalies
        :param random_state: Random seed for reproducibility
        :param normalize_features: Whether to standardize numeric features
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.normalize_features = normalize_features
        self.scaler = None

    def fit_predict(self, feature_df, feature_cols=None):
        """
        Fit the Isolation Forest and predict anomalies.
        Returns the dataframe with 'anomaly' and 'anomaly_score' columns.
        """
        if feature_cols is None:
            # Use all numeric columns except instrument/timestamp identifiers
            feature_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
            feature_cols = [c for c in feature_cols if c not in ["window_start", "window_end"]]

        X = feature_df[feature_cols].values

        # Optional normalization
        if self.normalize_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        df_copy = feature_df.copy()
        df_copy["anomaly"] = self.model.fit_predict(X)
        df_copy["anomaly_score"] = self.model.decision_function(X)

        return df_copy

    def detect_per_instrument(self, feature_df, instrument_col="Instrument Code", feature_cols=None, verbose=True):
        """
        Run anomaly detection per instrument and return combined results.
        Optionally prints anomaly counts per instrument.
        """
        result_list = []
        instruments = feature_df[instrument_col].unique()

        for inst in instruments:
            inst_df = feature_df[feature_df[instrument_col] == inst].copy()
            inst_df = self.fit_predict(inst_df, feature_cols)
            if verbose:
                num_anomalies = (inst_df["anomaly"] == -1).sum()
                print(f"Instrument {inst}: {num_anomalies} anomalies out of {len(inst_df)} rows")
            result_list.append(inst_df)

        return pd.concat(result_list, ignore_index=True)

    def get_anomalies(self, feature_df):
        """Return only rows flagged as anomalies"""
        return feature_df[feature_df["anomaly"] == -1]
