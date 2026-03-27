"""
model.py
Random Forest classifier with Leave-One-Subject-Out (LOSO) evaluation.

This module provides:
- train_and_evaluate_loso(): Full LOSO cross-validation across all subjects
- train_model(): Train a single RF model on given data
- predict(): Generate predictions with confidence scores
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from feature_extraction import get_feature_columns


def train_and_evaluate_loso(
    features_df: pd.DataFrame,
    binary: bool = True,
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict:
    """
    Leave-One-Subject-Out cross-validation.

    Args:
        features_df: DataFrame with features, 'label', and 'subject' columns
        binary: If True, map labels to Stress(1) vs Non-Stress(0)
        n_estimators: Number of trees in Random Forest
        random_state: Random seed

    Returns:
        dict with overall metrics and per-subject results
    """
    df = features_df.copy()
    feature_cols = get_feature_columns(df)

    if binary:
        # Stress (label=2) vs Non-Stress (label in {1, 3})
        df["target"] = (df["label"] == 2).astype(int)
        target_names = ["Non-Stress", "Stress"]
    else:
        df["target"] = df["label"]
        target_names = ["Baseline", "Stress", "Amusement"]

    subjects = sorted(df["subject"].unique())
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    per_subject = []

    for test_subject in subjects:
        train_mask = df["subject"] != test_subject
        test_mask = df["subject"] == test_subject

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "target"].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, "target"].values

        if len(X_test) == 0 or len(np.unique(y_train)) < 2:
            continue

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            max_depth=10,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        per_subject.append({
            "subject": test_subject,
            "accuracy": acc,
            "f1_weighted": f1,
            "n_test_windows": len(y_test),
            "stress_ratio": np.mean(y_test == 1) if binary else None,
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred)
    report = classification_report(
        all_y_true, all_y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    return {
        "overall_accuracy": overall_acc,
        "overall_f1_weighted": overall_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "per_subject": per_subject,
        "target_names": target_names,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
        "all_y_proba": all_y_proba,
    }


def train_model(
    features_df: pd.DataFrame,
    binary: bool = True,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple:
    """
    Train a single RF model on all data.
    Returns (model, feature_columns, label_encoder_info).
    """
    df = features_df.copy()
    feature_cols = get_feature_columns(df)

    if binary:
        df["target"] = (df["label"] == 2).astype(int)
    else:
        df["target"] = df["label"]

    X = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["target"].values

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        max_depth=10,
        n_jobs=-1,
    )
    clf.fit(X, y)

    return clf, feature_cols


def predict_windows(
    model, feature_cols: list, features_df: pd.DataFrame, binary: bool = True
) -> pd.DataFrame:
    """
    Generate predictions for windows with confidence scores.
    """
    df = features_df.copy()
    X = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    df["prediction"] = predictions
    if binary:
        df["stress_probability"] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        df["prediction_label"] = df["prediction"].map({0: "Non-Stress", 1: "Stress"})
    else:
        df["prediction_label"] = df["prediction"].map(
            {1: "Baseline", 2: "Stress", 3: "Amusement"}
        )
        for i, name in enumerate(model.classes_):
            df[f"prob_class_{name}"] = probabilities[:, i]

    # Confidence: max probability
    df["confidence"] = np.max(probabilities, axis=1)

    return df
