"""
data_generator.py
WESAD data loader with synthetic fallback.

If real WESAD pickle files are available in a 'wesad_data/' folder,
they are loaded and wrist signals (BVP, EDA, TEMP, ACC) are extracted.
Otherwise, WESAD-inspired synthetic data is generated for demonstration.

Disclaimer: Synthetic data is generated to approximate the statistical
properties reported in the WESAD paper (Schmidt et al., 2018) and is
intended solely for prototype demonstration, not clinical evaluation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# WESAD sampling rates for wrist-worn Empatica E4
SAMPLING_RATES = {
    "BVP": 64,    # Hz
    "EDA": 4,     # Hz
    "TEMP": 4,    # Hz
    "ACC": 32,    # Hz (3-axis)
}

# Condition labels used in WESAD
LABEL_MAP = {
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
}

# Duration per condition segment in seconds (for synthetic generation)
SEGMENT_DURATIONS = {
    1: 1200,  # 20 min baseline
    2: 600,   # 10 min stress (TSST)
    3: 390,   # 6.5 min amusement
}


def load_wesad_subject(pkl_path: str) -> dict | None:
    """Load a single WESAD subject pickle file and extract wrist data."""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        wrist = data["signal"]["wrist"]
        labels = data["label"]

        return {
            "BVP": wrist["BVP"].flatten(),
            "EDA": wrist["EDA"].flatten(),
            "TEMP": wrist["TEMP"].flatten(),
            "ACC_x": wrist["ACC"][:, 0],
            "ACC_y": wrist["ACC"][:, 1],
            "ACC_z": wrist["ACC"][:, 2],
            "labels": labels.flatten(),
        }
    except Exception as e:
        print(f"Warning: Could not load {pkl_path}: {e}")
        return None


def load_wesad_dataset(data_dir: str) -> dict:
    """
    Attempt to load real WESAD data from data_dir.
    Expected structure: data_dir/S2/S2.pkl, data_dir/S3/S3.pkl, etc.
    Returns dict mapping subject_id -> subject data dict.
    """
    subjects = {}
    data_path = Path(data_dir)

    if not data_path.exists():
        return subjects

    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("S"):
            pkl_file = subdir / f"{subdir.name}.pkl"
            if pkl_file.exists():
                subject_data = load_wesad_subject(str(pkl_file))
                if subject_data is not None:
                    subjects[subdir.name] = subject_data

    return subjects


def _generate_synthetic_signal(
    n_samples: int,
    condition: int,
    signal_type: str,
    rng: np.random.Generator,
    subject_variation: float = 0.0,
) -> np.ndarray:
    """
    Generate a single synthetic signal segment based on WESAD-inspired
    statistical properties for a given condition.

    subject_variation: a per-subject offset [-1, 1] that shifts signal
    properties to simulate individual physiological differences.
    This creates realistic cross-subject variability where some subjects
    show weaker stress responses, making classification harder.
    """
    t = np.arange(n_samples)
    sv = subject_variation  # shorthand

    if signal_type == "BVP":
        # BVP: quasi-periodic pulse signal
        # Subject variation affects resting HR and stress response magnitude
        base_hr = 72 + sv * 12  # resting HR varies 60-84 across people
        if condition == 2:  # Stress: HR increase varies by person
            freq = (base_hr + 15 + sv * 8) / 60  # some people react more
            amplitude = 80 + rng.normal(0, 15)
        elif condition == 3:  # Amusement
            freq = (base_hr + 5 + sv * 3) / 60
            amplitude = 100 + rng.normal(0, 15)
        else:  # Baseline
            freq = base_hr / 60
            amplitude = 110 + rng.normal(0, 15)

        sr = SAMPLING_RATES["BVP"]
        # Add frequency jitter (HR variability)
        phase_noise = np.cumsum(rng.normal(0, 0.02, n_samples))
        signal = amplitude * np.sin(2 * np.pi * freq * t / sr + phase_noise)
        signal += rng.normal(0, 20, n_samples)  # heavier sensor noise
        # Add occasional motion artefacts
        n_artefacts = rng.integers(2, 8)
        for _ in range(n_artefacts):
            pos = rng.integers(0, max(1, n_samples - 50))
            length = rng.integers(10, 50)
            end_pos = min(pos + length, n_samples)
            signal[pos:end_pos] += rng.normal(0, 40, end_pos - pos)
        return signal

    elif signal_type == "EDA":
        # EDA: tonic level + phasic responses
        # Large individual variation in baseline EDA
        base_eda = 3.0 + sv * 4.0  # range ~-1 to 7 µS baseline
        if condition == 2:  # Stress: elevated SCL (but amount varies)
            tonic = base_eda + 3.0 + rng.normal(0, 2.0)
            n_scr = rng.integers(5, 25)
        elif condition == 3:  # Amusement
            tonic = base_eda + 1.0 + rng.normal(0, 1.5)
            n_scr = rng.integers(3, 12)
        else:  # Baseline
            tonic = base_eda + rng.normal(0, 1.0)
            n_scr = rng.integers(1, 8)

        signal = np.full(n_samples, max(tonic, 0.1))
        # Add slow drift
        drift = np.cumsum(rng.normal(0, 0.005, n_samples))
        signal += drift
        # Add skin conductance responses (SCRs)
        for _ in range(n_scr):
            onset = rng.integers(0, max(1, n_samples - 40))
            amplitude_scr = rng.uniform(0.1, 2.0)
            rise = np.linspace(0, amplitude_scr, rng.integers(10, 25))
            decay = np.linspace(amplitude_scr, 0, rng.integers(20, 60))
            scr = np.concatenate([rise, decay])
            end = min(onset + len(scr), n_samples)
            signal[onset:end] += scr[: end - onset]

        signal += rng.normal(0, 0.15, n_samples)
        return np.maximum(signal, 0)

    elif signal_type == "TEMP":
        # Skin temperature — very high individual variation
        base_temp = 33.0 + sv * 2.0  # range ~31-35°C
        if condition == 2:  # Stress: slight decrease (but noisy)
            base = base_temp - 0.3 + rng.normal(0, 0.8)
            drift = np.linspace(0, rng.normal(-0.2, 0.15), n_samples)
        elif condition == 3:  # Amusement
            base = base_temp + rng.normal(0, 0.6)
            drift = np.linspace(0, rng.normal(0.1, 0.1), n_samples)
        else:  # Baseline
            base = base_temp + rng.normal(0, 0.5)
            drift = np.linspace(0, rng.normal(0.0, 0.1), n_samples)

        signal = base + drift + rng.normal(0, 0.08, n_samples)
        return signal

    elif signal_type.startswith("ACC"):
        # Accelerometer: gravity + activity
        # Activity level varies across subjects
        base_activity = 0.08 + abs(sv) * 0.05
        if condition == 2:  # Stress: slightly more fidgeting
            activity_std = base_activity + 0.06 + rng.uniform(0, 0.08)
        elif condition == 3:  # Amusement: some movement
            activity_std = base_activity + 0.04 + rng.uniform(0, 0.06)
        else:  # Baseline
            activity_std = base_activity + rng.uniform(0, 0.03)

        activity = rng.normal(0, activity_std, n_samples)

        # Occasional larger movements (fidgeting, posture shifts)
        n_movements = rng.integers(1, 6)
        for _ in range(n_movements):
            pos = rng.integers(0, max(1, n_samples - 20))
            length = rng.integers(5, 20)
            end_pos = min(pos + length, n_samples)
            activity[pos:end_pos] += rng.normal(0, 0.3, end_pos - pos)

        # Gravity component depends on axis
        if signal_type == "ACC_x":
            gravity = 0.0
        elif signal_type == "ACC_y":
            gravity = 0.0
        else:  # ACC_z
            gravity = 1.0

        return gravity + activity

    return rng.normal(0, 1, n_samples)


def generate_synthetic_subject(
    subject_id: int, seed: int | None = None
) -> dict:
    """
    Generate synthetic wrist data for one subject across all conditions.
    Returns a dict with concatenated signals and label array.

    Each subject gets a unique physiological profile via subject_variation,
    which shifts signal properties to create realistic inter-subject
    variability — the key challenge in real-world stress detection.
    """
    rng = np.random.default_rng(seed)

    all_signals = {
        "BVP": [], "EDA": [], "TEMP": [],
        "ACC_x": [], "ACC_y": [], "ACC_z": [],
    }
    all_labels = []

    # Add individual variation via seed offset
    subject_rng = np.random.default_rng(seed + subject_id * 1000 if seed else subject_id * 42)

    # Per-subject physiological variation: range [-1, 1]
    # This simulates that different people have different baseline physiology
    # and different stress response magnitudes
    subject_variation = subject_rng.uniform(-1.0, 1.0)

    for condition in [1, 2, 3]:
        duration = SEGMENT_DURATIONS[condition]

        for sig_name, sr in [
            ("BVP", SAMPLING_RATES["BVP"]),
            ("EDA", SAMPLING_RATES["EDA"]),
            ("TEMP", SAMPLING_RATES["TEMP"]),
        ]:
            n_samples = duration * sr
            sig = _generate_synthetic_signal(
                n_samples, condition, sig_name, subject_rng, subject_variation
            )
            all_signals[sig_name].append(sig)

        # ACC at 32 Hz, 3 axes
        n_acc = duration * SAMPLING_RATES["ACC"]
        for axis in ["ACC_x", "ACC_y", "ACC_z"]:
            sig = _generate_synthetic_signal(
                n_acc, condition, axis, subject_rng, subject_variation
            )
            all_signals[axis].append(sig)

        # Labels at 700 Hz in real WESAD, but we store per-second for simplicity
        # We'll expand labels to match the lowest SR (4 Hz) for alignment
        n_label_samples = duration * 4  # align with EDA/TEMP rate
        all_labels.append(np.full(n_label_samples, condition))

    result = {}
    for key in all_signals:
        result[key] = np.concatenate(all_signals[key])
    result["labels"] = np.concatenate(all_labels)

    return result


def generate_synthetic_dataset(n_subjects: int = 30, seed: int = 42) -> dict:
    """
    Generate synthetic wrist data for multiple subjects.
    Returns dict mapping subject_id (e.g. 'S1') -> subject data.
    Sequential numbering S1-S15 for clean presentation.
    """
    subjects = {}

    for sid in range(1, n_subjects + 1):
        subject_key = f"S{sid}"
        subjects[subject_key] = generate_synthetic_subject(sid, seed=seed)

    return subjects


def get_dataset(wesad_dir: str = "wesad_data", n_synthetic: int = 30) -> tuple[dict, bool]:
    """
    Try to load real WESAD data. If not available, generate synthetic data.
    Returns (subjects_dict, is_real_data).
    """
    real_data = load_wesad_dataset(wesad_dir)
    if len(real_data) >= 2:
        return real_data, True
    else:
        return generate_synthetic_dataset(n_subjects=n_synthetic), False


def subject_to_dataframe(subject_data: dict, subject_id: str) -> pd.DataFrame:
    """
    Convert a subject's data dict into a pandas DataFrame aligned at 4 Hz
    (the lowest common rate for EDA/TEMP). BVP and ACC are downsampled.
    """
    n_samples = len(subject_data["EDA"])  # 4 Hz reference length
    labels = subject_data["labels"][:n_samples]

    # Downsample BVP from 64 Hz to 4 Hz (take every 16th sample)
    bvp_full = subject_data["BVP"]
    bvp_ds = bvp_full[::16][:n_samples]

    # Downsample ACC from 32 Hz to 4 Hz (take every 8th sample)
    acc_x_ds = subject_data["ACC_x"][::8][:n_samples]
    acc_y_ds = subject_data["ACC_y"][::8][:n_samples]
    acc_z_ds = subject_data["ACC_z"][::8][:n_samples]

    # TEMP and EDA are already at 4 Hz
    eda = subject_data["EDA"][:n_samples]
    temp = subject_data["TEMP"][:n_samples]

    # Compute ACC magnitude
    acc_mag = np.sqrt(acc_x_ds**2 + acc_y_ds**2 + acc_z_ds**2)

    df = pd.DataFrame({
        "BVP": bvp_ds,
        "EDA": eda,
        "TEMP": temp,
        "ACC_x": acc_x_ds,
        "ACC_y": acc_y_ds,
        "ACC_z": acc_z_ds,
        "ACC_mag": acc_mag,
        "label": labels,
        "subject": subject_id,
    })

    # Add time column in seconds
    df["time_s"] = np.arange(len(df)) / 4.0

    # Add condition name
    df["condition"] = df["label"].map(LABEL_MAP).fillna("Other")

    return df
