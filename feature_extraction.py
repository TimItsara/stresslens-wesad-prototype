"""
feature_extraction.py
Statistical feature extraction from segmented windows.

For each window and each signal channel, the following features are computed:
- mean, std, min, max, range
- median, IQR
- slope (linear trend)
- number of zero-crossings (of the de-meaned signal)

These features are consistent with common practice in wearable stress
detection literature (Schmidt et al., 2018; Gjoreski et al., 2017).
"""

import numpy as np
import pandas as pd


def extract_window_features(window_data: dict) -> dict:
    """
    Extract statistical features from a single window's signal data.

    Args:
        window_data: dict mapping signal_name -> np.ndarray

    Returns:
        dict of feature_name -> value
    """
    features = {}

    for signal_name, signal_values in window_data.items():
        if len(signal_values) == 0:
            continue

        prefix = signal_name

        # Basic statistics
        features[f"{prefix}_mean"] = np.mean(signal_values)
        features[f"{prefix}_std"] = np.std(signal_values)
        features[f"{prefix}_min"] = np.min(signal_values)
        features[f"{prefix}_max"] = np.max(signal_values)
        features[f"{prefix}_range"] = np.ptp(signal_values)
        features[f"{prefix}_median"] = np.median(signal_values)

        # IQR
        q75, q25 = np.percentile(signal_values, [75, 25])
        features[f"{prefix}_iqr"] = q75 - q25

        # Linear slope (trend within window)
        x = np.arange(len(signal_values))
        if len(signal_values) > 1:
            slope = np.polyfit(x, signal_values, 1)[0]
        else:
            slope = 0.0
        features[f"{prefix}_slope"] = slope

        # Zero-crossings of de-meaned signal
        demeaned = signal_values - np.mean(signal_values)
        zero_crossings = np.sum(np.diff(np.sign(demeaned)) != 0)
        features[f"{prefix}_zc"] = zero_crossings

    return features


def extract_features_from_windows(windows: list[dict]) -> pd.DataFrame:
    """
    Extract features from all windows and return a DataFrame.

    Args:
        windows: List of window dicts from preprocessing.segment_windows()

    Returns:
        DataFrame where each row is a window with extracted features,
        plus metadata columns (label, subject, start_time, end_time).
    """
    rows = []
    for w in windows:
        features = extract_window_features(w["data"])
        features["label"] = w["label"]
        features["subject"] = w["subject"]
        features["start_time"] = w["start_time"]
        features["end_time"] = w["end_time"]
        rows.append(features)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return only the feature columns (exclude metadata)."""
    metadata = {"label", "subject", "start_time", "end_time"}
    return [c for c in df.columns if c not in metadata]
