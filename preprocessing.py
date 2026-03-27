"""
preprocessing.py
Signal cleaning, normalisation, and sliding window segmentation
for wrist-worn wearable stress detection.

Preprocessing steps:
1. Bandpass / lowpass filtering to remove high-frequency noise
2. Z-score normalisation per subject
3. Sliding window segmentation (default 60s windows, 50% overlap)
4. Label assignment per window (majority vote)
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal


def bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    """Apply a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    if low >= high:
        return data
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return scipy_signal.filtfilt(b, a, data)


def lowpass_filter(
    data: np.ndarray, cutoff: float, fs: float, order: int = 4
) -> np.ndarray:
    """Apply a Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normalized_cutoff = min(cutoff / nyq, 0.999)
    b, a = scipy_signal.butter(order, normalized_cutoff, btype="low")
    return scipy_signal.filtfilt(b, a, data)


def z_normalise(data: np.ndarray) -> np.ndarray:
    """Z-score normalisation."""
    std = np.std(data)
    if std < 1e-8:
        return data - np.mean(data)
    return (data - np.mean(data)) / std


def clean_signals(df: pd.DataFrame, fs: float = 4.0) -> pd.DataFrame:
    """
    Apply appropriate filtering to each signal channel.
    Input df is at 4 Hz (downsampled).
    """
    cleaned = df.copy()

    # BVP: bandpass 0.5-2.0 Hz (pulse range)
    if "BVP" in cleaned.columns:
        cleaned["BVP"] = lowpass_filter(cleaned["BVP"].values, cutoff=1.8, fs=fs)

    # EDA: lowpass 1.0 Hz (remove high-freq noise, keep SCRs)
    if "EDA" in cleaned.columns:
        cleaned["EDA"] = lowpass_filter(cleaned["EDA"].values, cutoff=1.0, fs=fs)

    # TEMP: lowpass 0.5 Hz (very slow-changing signal)
    if "TEMP" in cleaned.columns:
        cleaned["TEMP"] = lowpass_filter(cleaned["TEMP"].values, cutoff=0.5, fs=fs)

    # ACC: lowpass 1.5 Hz (remove vibration noise)
    for col in ["ACC_x", "ACC_y", "ACC_z", "ACC_mag"]:
        if col in cleaned.columns:
            cleaned[col] = lowpass_filter(cleaned[col].values, cutoff=1.5, fs=fs)

    return cleaned


def normalise_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalise all signal columns per subject."""
    normalised = df.copy()
    signal_cols = ["BVP", "EDA", "TEMP", "ACC_x", "ACC_y", "ACC_z", "ACC_mag"]
    for col in signal_cols:
        if col in normalised.columns:
            normalised[col] = z_normalise(normalised[col].values)
    return normalised


def segment_windows(
    df: pd.DataFrame,
    window_size_s: float = 60.0,
    overlap: float = 0.5,
    fs: float = 4.0,
    min_label_ratio: float = 0.6,
) -> list[dict]:
    """
    Segment the dataframe into fixed-size sliding windows.

    Args:
        df: DataFrame with signal columns and 'label' column
        window_size_s: Window duration in seconds
        overlap: Overlap ratio (0.5 = 50% overlap)
        fs: Sampling rate in Hz
        min_label_ratio: Minimum proportion of dominant label to accept window

    Returns:
        List of dicts, each containing window data and metadata
    """
    window_samples = int(window_size_s * fs)
    step_samples = int(window_samples * (1 - overlap))

    signal_cols = ["BVP", "EDA", "TEMP", "ACC_x", "ACC_y", "ACC_z", "ACC_mag"]
    available_cols = [c for c in signal_cols if c in df.columns]

    windows = []
    n = len(df)

    for start in range(0, n - window_samples + 1, step_samples):
        end = start + window_samples
        window_df = df.iloc[start:end]

        # Majority vote for label
        labels_in_window = window_df["label"].values
        # Only keep windows with labels 1, 2, or 3
        valid_mask = np.isin(labels_in_window, [1, 2, 3])
        if valid_mask.sum() < min_label_ratio * window_samples:
            continue

        valid_labels = labels_in_window[valid_mask]
        unique, counts = np.unique(valid_labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]
        dominant_ratio = counts.max() / valid_mask.sum()

        if dominant_ratio < min_label_ratio:
            continue

        window_data = {}
        for col in available_cols:
            window_data[col] = window_df[col].values.copy()

        windows.append({
            "data": window_data,
            "label": int(dominant_label),
            "start_time": window_df["time_s"].iloc[0],
            "end_time": window_df["time_s"].iloc[-1],
            "subject": window_df["subject"].iloc[0],
            "dominant_ratio": dominant_ratio,
        })

    return windows


def preprocess_subject(
    df: pd.DataFrame, window_size_s: float = 60.0, overlap: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Full preprocessing pipeline for a single subject.

    Returns:
        - cleaned_df: Filtered signals
        - normalised_df: Z-normalised signals
        - windows: List of segmented window dicts
    """
    cleaned_df = clean_signals(df)
    normalised_df = normalise_signals(cleaned_df)
    windows = segment_windows(normalised_df, window_size_s, overlap)
    return cleaned_df, normalised_df, windows
