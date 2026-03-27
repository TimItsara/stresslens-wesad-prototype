"""
StressLens — WESAD-Guided Wearable Stress Monitoring Prototype
A proof-of-concept academic prototype for COMP8230.

This application demonstrates a wrist-based wearable stress monitoring
workflow using WESAD-guided data. It is an academic prototype for
monitoring and decision-support purposes, NOT a clinical diagnostic tool.
"""

import sys
import os
import re
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def natural_sort_key(s):
    """Sort S2, S3, ..., S10, S11 numerically instead of lexicographically."""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

from data_generator import get_dataset, subject_to_dataframe, LABEL_MAP
from preprocessing import preprocess_subject
from feature_extraction import extract_features_from_windows, get_feature_columns
from model import train_and_evaluate_loso, train_model, predict_windows

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="StressLens — Wearable Stress Monitoring",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Plotly template ──────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(color="#1a1a2e", size=13),
    colorway=["#4361ee", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"],
)


def apply_axis_spacing(fig):
    """Add spacing between axis title and tick labels for readability."""
    fig.update_yaxes(title_standoff=18)
    fig.update_xaxes(title_standoff=12)
    return fig

CONDITION_COLORS = {
    "Baseline": "#2ecc71",
    "Stress": "#e74c3c",
    "Amusement": "#3498db",
}


# ── Data Loading (cached) ────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    subjects, is_real = get_dataset()
    return subjects, is_real


@st.cache_data(show_spinner="Preprocessing signals...")
def preprocess_data(subject_id, _subjects):
    subject_data = _subjects[subject_id]
    df = subject_to_dataframe(subject_data, subject_id)
    cleaned_df, normalised_df, windows = preprocess_subject(df)
    return df, cleaned_df, normalised_df, windows


@st.cache_data(show_spinner="Extracting features from all subjects...")
def get_all_features(_subjects):
    all_features = []
    for sid in sorted(_subjects.keys(), key=natural_sort_key):
        df = subject_to_dataframe(_subjects[sid], sid)
        _, _, windows = preprocess_subject(df)
        feat_df = extract_features_from_windows(windows)
        all_features.append(feat_df)
    return pd.concat(all_features, ignore_index=True)


@st.cache_data(show_spinner="Running LOSO evaluation...")
def run_loso_evaluation(_features_df):
    return train_and_evaluate_loso(_features_df)


# ── Load Data ────────────────────────────────────────────────────────
subjects, is_real_data = load_data()
subject_ids = sorted(subjects.keys(), key=natural_sort_key)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("StressLens")
    st.caption("Wearable Stress Monitoring Prototype")
    st.divider()

    data_label = "WESAD Benchmark Data" if is_real_data else "WESAD-Inspired Demo Data"
    st.info(f"**Data source:** {data_label}", icon="📂")
    st.caption(
        "Based on [WESAD (Wearable Stress and Affect Detection)]"
        "(https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) "
        "— Schmidt et al., 2018"
    )

    selected_subject = st.selectbox(
        "Select Subject / Session",
        subject_ids,
        index=0,
        help="Choose a subject to examine their wearable sensor data and stress inference results.",
    )

    st.divider()

    # Methodology summary — expandable
    with st.expander("Methodology", expanded=False):
        st.markdown("""
**Sensing:** Wrist-worn (Empatica E4-style)
- Blood Volume Pulse (BVP)
- Electrodermal Activity (EDA)
- Skin Temperature
- 3-axis Accelerometer

**Pipeline:**
1. Lowpass filtering per channel
2. Z-score normalisation
3. 60s sliding windows (50% overlap)
4. 63 statistical features per window
5. Random Forest classifier

**Evaluation:**
- Leave-One-Subject-Out (LOSO)
- Binary: Stress vs Non-Stress
- Metrics: Accuracy, F1, Confusion Matrix
        """)

    with st.expander("About", expanded=False):
        st.markdown(
            "Academic proof-of-concept for **COMP8230** "
            "(Mining Unstructured Data), Macquarie University.\n\n"
            "This is a monitoring and decision-support "
            "demonstration — **not** a clinical diagnostic tool.\n\n"
            "---\n"
            "**Developed by**\n\n"
            "Itsara Timaroon (48572918)\n\n"
            "School of Computing, Macquarie University\n\n"
            "Semester 1, 2026"
        )

# ── Main Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Signal Viewer",
    "🔧 Preprocessing",
    "🧠 Stress Inference",
    "📈 Trends & Episodes",
    "📋 Evaluation (LOSO)",
])

# Preprocess selected subject
raw_df, cleaned_df, normalised_df, windows = preprocess_data(selected_subject, subjects)

# Pre-compute model (used by multiple tabs)
all_features = get_all_features(subjects)
trained_model, feature_cols = train_model(all_features)
subject_features = extract_features_from_windows(windows)


# ══════════════════════════════════════════════════════════════════════
# TAB 1: Signal Viewer
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Raw Sensor Signals")
    st.markdown(
        f"**Subject {selected_subject}** — "
        "Wrist-worn wearable signals: BVP, EDA, Skin Temperature, Accelerometer"
    )

    if not is_real_data:
        st.warning(
            "**Demonstration data:** This prototype uses WESAD-inspired synthetic data. "
            "Signal shapes approximate the statistical properties reported in the WESAD "
            "benchmark (Schmidt et al., 2018) but are not recorded from real participants.",
            icon="ℹ️",
        )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    total_duration = raw_df["time_s"].max()
    with col1:
        st.metric("Duration", f"{total_duration/60:.1f} min")
    with col2:
        stress_pct = (raw_df["label"] == 2).mean() * 100
        st.metric("Stress Segments", f"{stress_pct:.1f}%")
    with col3:
        st.metric("Sampling Rate", "4 Hz (aligned)")
    with col4:
        st.metric("Total Samples", f"{len(raw_df):,}")

    st.divider()

    # Signal selector — let user choose which signals to view
    signals_available = [
        ("BVP", "Blood Volume Pulse", "mV"),
        ("EDA", "Electrodermal Activity", "µS"),
        ("TEMP", "Skin Temperature", "°C"),
        ("ACC_mag", "Accelerometer Magnitude", "g"),
    ]

    for sig_col, sig_name, unit in signals_available:
        fig = px.line(
            raw_df, x="time_s", y=sig_col,
            color="condition",
            color_discrete_map=CONDITION_COLORS,
            labels={"time_s": "Time (s)", sig_col: f"{sig_name} ({unit})"},
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=sig_name, font=dict(size=15)),
            height=230,
            margin=dict(l=65, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            showlegend=(sig_col == "BVP"),
        )
        st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

    # Signal statistics per condition
    st.divider()
    st.subheader("Signal Statistics by Condition")

    signal_cols = ["BVP", "EDA", "TEMP", "ACC_mag"]
    stats_data = []
    for cond_id, cond_name in LABEL_MAP.items():
        cond_df = raw_df[raw_df["label"] == cond_id]
        for col in signal_cols:
            stats_data.append({
                "Condition": cond_name,
                "Signal": col,
                "Mean": round(cond_df[col].mean(), 3),
                "Std": round(cond_df[col].std(), 3),
                "Min": round(cond_df[col].min(), 3),
                "Max": round(cond_df[col].max(), 3),
            })
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: Preprocessing
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Preprocessing Pipeline")
    st.markdown("Signal cleaning, normalisation, and window segmentation applied to raw wearable data.")

    # Pipeline steps in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Filtering")
        st.markdown(
            "- **BVP:** lowpass 1.8 Hz\n"
            "- **EDA:** lowpass 1.0 Hz\n"
            "- **TEMP:** lowpass 0.5 Hz\n"
            "- **ACC:** lowpass 1.5 Hz\n\n"
            "Removes high-frequency noise while preserving physiological patterns."
        )
    with col2:
        st.subheader("2. Normalisation")
        st.markdown(
            "- Z-score normalisation\n"
            "- Per-subject, per-channel\n"
            "- Removes amplitude bias\n"
            "- Enables cross-subject comparison"
        )
    with col3:
        st.subheader("3. Windowing")
        st.markdown(
            "- 60-second sliding windows\n"
            "- 50% overlap\n"
            "- Majority-vote labelling\n"
            f"- **{len(windows)} windows** extracted"
        )

    st.divider()

    # Before / After comparison
    st.subheader("Raw vs. Cleaned Signal Comparison")
    compare_signal = st.selectbox(
        "Select signal to compare",
        ["EDA", "BVP", "TEMP", "ACC_mag"],
        index=0,
    )

    show_samples = min(len(raw_df), 4 * 60 * 5)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=("Raw Signal", "After Filtering + Normalisation"),
    )
    fig.add_trace(
        go.Scatter(
            x=raw_df["time_s"].iloc[:show_samples],
            y=raw_df[compare_signal].iloc[:show_samples],
            mode="lines", name="Raw", line=dict(color="#e74c3c", width=1.2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=normalised_df["time_s"].iloc[:show_samples],
            y=normalised_df[compare_signal].iloc[:show_samples],
            mode="lines", name="Cleaned + Normalised", line=dict(color="#2ecc71", width=1.2),
        ),
        row=2, col=1,
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=420, margin=dict(l=65, r=10, t=35, b=10))
    st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

    # Signal correlation heatmap
    st.divider()
    st.subheader("Inter-Signal Correlation")
    st.markdown(
        "Correlation between normalised signal channels. "
        "Understanding how signals relate helps interpret multimodal stress patterns."
    )

    corr_cols = ["BVP", "EDA", "TEMP", "ACC_mag"]
    corr_matrix = normalised_df[corr_cols].corr()

    fig = px.imshow(
        corr_matrix.round(3),
        x=corr_cols, y=corr_cols,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=350, margin=dict(l=65, r=10, t=10, b=10))
    st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

    # Window segmentation summary
    st.divider()
    st.subheader("Window Segmentation Summary")
    if windows:
        label_counts = pd.Series([w["label"] for w in windows]).map(LABEL_MAP).value_counts()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(label_counts.rename("Windows").to_frame(), use_container_width=True)
        with col2:
            fig = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                color=label_counts.index,
                color_discrete_map=CONDITION_COLORS,
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Window Distribution by Condition", font=dict(size=14)),
                height=280,
                margin=dict(l=65, r=10, t=40, b=10),
            )
            st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3: Stress Inference
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Stress Inference Results")
    st.markdown(
        f"**Subject {selected_subject}** — "
        "Random Forest binary classification (Stress vs Non-Stress)"
    )

    st.info(
        "**Interpretation note:** Stress likelihood scores are model estimates based on "
        "physiological signal patterns. They should be interpreted as indicative trends, "
        "not confirmed stress states. This is a monitoring aid, not a diagnosis.",
        icon="💡",
    )

    if len(subject_features) > 0:
        results_df = predict_windows(trained_model, feature_cols, subject_features)

        # Timeline of stress probability
        st.subheader("Stress Likelihood Over Time")
        fig = go.Figure()

        colors = ["#e74c3c" if p > 0.5 else "#2ecc71" for p in results_df["stress_probability"]]

        fig.add_trace(go.Scatter(
            x=results_df["start_time"] / 60,
            y=results_df["stress_probability"],
            mode="lines+markers",
            name="Stress Likelihood",
            line=dict(color="#4361ee", width=2.5),
            marker=dict(size=7, color=colors, line=dict(width=1, color="#ffffff")),
            fill="tozeroy",
            fillcolor="rgba(67, 97, 238, 0.08)",
        ))

        fig.add_hline(
            y=0.5, line_dash="dash", line_color="#aaaaaa", line_width=1.5,
            annotation_text="Decision threshold (0.5)",
            annotation_font=dict(color="#666666", size=12),
        )

        fig.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Time (minutes)",
            yaxis_title="Stress Likelihood",
            yaxis=dict(range=[0, 1.05]),
            height=370,
            margin=dict(l=65, r=10, t=10, b=10),
        )
        st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        n_stress = (results_df["prediction"] == 1).sum()
        n_total = len(results_df)
        avg_conf = results_df["confidence"].mean()
        max_stress_prob = results_df["stress_probability"].max()

        with col1:
            st.metric("Windows Analysed", n_total)
        with col2:
            st.metric("Stress-Flagged Windows", f"{n_stress} / {n_total}")
        with col3:
            st.metric("Avg. Model Confidence", f"{avg_conf:.1%}")
        with col4:
            st.metric("Peak Stress Likelihood", f"{max_stress_prob:.1%}")

        # Confidence distribution
        st.divider()
        st.subheader("Prediction Confidence Distribution")
        col_left, col_right = st.columns(2)

        with col_left:
            fig = px.histogram(
                results_df, x="stress_probability", nbins=20,
                color_discrete_sequence=["#4361ee"],
                labels={"stress_probability": "Stress Likelihood"},
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="#aaaaaa")
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Stress Likelihood Distribution", font=dict(size=14)),
                height=300, margin=dict(l=65, r=10, t=40, b=10),
                yaxis_title="Window Count",
            )
            st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

        with col_right:
            fig = px.histogram(
                results_df, x="confidence", nbins=20,
                color_discrete_sequence=["#2ecc71"],
                labels={"confidence": "Model Confidence"},
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Model Confidence Distribution", font=dict(size=14)),
                height=300, margin=dict(l=65, r=10, t=40, b=10),
                yaxis_title="Window Count",
            )
            st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

        # Per-window detail table
        st.divider()
        st.subheader("Per-Window Predictions")
        display_cols = ["start_time", "end_time", "prediction_label",
                        "stress_probability", "confidence", "label"]
        display_df = results_df[display_cols].copy()
        display_df["start_time"] = (display_df["start_time"] / 60).round(1)
        display_df["end_time"] = (display_df["end_time"] / 60).round(1)
        display_df["stress_probability"] = display_df["stress_probability"].round(3)
        display_df["confidence"] = display_df["confidence"].round(3)
        display_df["ground_truth"] = display_df["label"].map(LABEL_MAP)
        display_df = display_df.drop(columns=["label"])
        display_df.columns = ["Start (min)", "End (min)", "Prediction",
                              "Stress Likelihood", "Confidence", "Ground Truth"]

        st.dataframe(display_df, use_container_width=True, height=300)
    else:
        st.warning("No windows available for this subject.")


# ══════════════════════════════════════════════════════════════════════
# TAB 4: Trends & Episodes
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Stress Trend & Episode Summary")
    st.markdown("Session-level stress pattern analysis with self-management guidance.")

    if len(subject_features) > 0:
        results_df = predict_windows(trained_model, feature_cols, subject_features)

        # Overview metrics at the top
        col1, col2, col3 = st.columns(3)
        n_stress = (results_df["prediction"] == 1).sum()
        n_total = len(results_df)
        stress_ratio = n_stress / n_total if n_total > 0 else 0

        with col1:
            st.metric("Session Duration", f"{total_duration/60:.1f} min")
        with col2:
            st.metric("Stress-Flagged", f"{stress_ratio:.0%} of session")
        with col3:
            trend = "Stable" if np.std(results_df["stress_probability"]) < 0.15 else "Variable"
            st.metric("Stress Pattern", trend)

        st.divider()

        # Stress episode detection
        st.subheader("Detected Stress Episodes")
        stress_windows = results_df[results_df["prediction"] == 1].copy()

        if len(stress_windows) > 0:
            episodes = []
            current_start = None
            current_end = None
            current_peak = 0

            for _, row in stress_windows.iterrows():
                if current_start is None:
                    current_start = row["start_time"]
                    current_end = row["end_time"]
                    current_peak = row["stress_probability"]
                elif row["start_time"] - current_end <= 60:
                    current_end = row["end_time"]
                    current_peak = max(current_peak, row["stress_probability"])
                else:
                    episodes.append({
                        "Start (min)": round(current_start / 60, 1),
                        "End (min)": round(current_end / 60, 1),
                        "Duration (min)": round((current_end - current_start) / 60, 1),
                        "Peak Likelihood": round(current_peak, 3),
                    })
                    current_start = row["start_time"]
                    current_end = row["end_time"]
                    current_peak = row["stress_probability"]

            if current_start is not None:
                episodes.append({
                    "Start (min)": round(current_start / 60, 1),
                    "End (min)": round(current_end / 60, 1),
                    "Duration (min)": round((current_end - current_start) / 60, 1),
                    "Peak Likelihood": round(current_peak, 3),
                })

            episodes_df = pd.DataFrame(episodes)
            episodes_df.index = [f"Episode {i+1}" for i in range(len(episodes_df))]

            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(episodes_df, use_container_width=True)
            with col2:
                st.metric("Total Episodes", len(episodes))
                total_stress_min = episodes_df["Duration (min)"].sum()
                st.metric("Total Stress Duration", f"{total_stress_min:.1f} min")

            # Episode timeline visualisation
            st.divider()
            st.subheader("Session Timeline")

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[total_duration / 60], y=["Session"],
                orientation="h", marker_color="#2ecc71",
                name="Non-Stress", opacity=0.25,
            ))

            for i, ep in enumerate(episodes):
                fig.add_trace(go.Bar(
                    x=[ep["Duration (min)"]],
                    y=["Session"],
                    base=[ep["Start (min)"]],
                    orientation="h",
                    marker_color="#e74c3c",
                    name="Stress Episode" if i == 0 else None,
                    showlegend=(i == 0),
                    opacity=0.75,
                ))

            fig.update_layout(
                **PLOTLY_LAYOUT,
                barmode="overlay",
                height=130,
                margin=dict(l=65, r=10, t=10, b=10),
                xaxis_title="Time (minutes)",
                legend=dict(orientation="h", yanchor="bottom", y=1.05),
            )
            st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

            # Stress likelihood rolling trend
            st.divider()
            st.subheader("Stress Trend (Rolling Average)")
            window_size = min(5, len(results_df))
            results_df["rolling_stress"] = (
                results_df["stress_probability"]
                .rolling(window=window_size, min_periods=1, center=True)
                .mean()
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results_df["start_time"] / 60,
                y=results_df["stress_probability"],
                mode="markers",
                name="Per-Window",
                marker=dict(size=6, color="#4361ee", opacity=0.4),
            ))
            fig.add_trace(go.Scatter(
                x=results_df["start_time"] / 60,
                y=results_df["rolling_stress"],
                mode="lines",
                name=f"Rolling Mean ({window_size}-window)",
                line=dict(color="#e74c3c", width=3),
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#aaaaaa", line_width=1)
            fig.update_layout(
                **PLOTLY_LAYOUT,
                xaxis_title="Time (minutes)",
                yaxis_title="Stress Likelihood",
                yaxis=dict(range=[0, 1.05]),
                height=300,
                margin=dict(l=65, r=10, t=10, b=10),
            )
            st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

        else:
            st.success("No stress episodes detected in this session.", icon="✅")

        # Self-management suggestions
        st.divider()
        st.subheader("Self-Management Suggestions")

        st.info(
            "These suggestions are general wellbeing guidance based on detected "
            "stress patterns. They are **not** medical advice. Users experiencing "
            "persistent stress should consult appropriate support services.",
            icon="📋",
        )

        if stress_ratio > 0.5:
            st.error(
                "**High stress detected** across a large portion of this session. "
                "Consider taking a longer break or changing your current activity.",
                icon="🔴",
            )
            st.markdown(
                "- Review your recent schedule for sustained high-demand periods.\n"
                "- Consider adjusting workload distribution or taking a rest day.\n"
                "- Try a brief relaxation exercise (e.g., deep breathing for 2 minutes)."
            )
        elif stress_ratio > 0.2:
            st.warning(
                "**Moderate stress patterns** detected. "
                "A short break (5-10 minutes) may help.",
                icon="🟡",
            )
            st.markdown(
                "- Consider logging what you were doing during flagged periods "
                "to identify recurring triggers.\n"
                "- A brief walk or change of environment can help reset."
            )
        else:
            st.success(
                "**Low stress levels** observed in this session. "
                "Current activity patterns appear manageable.",
                icon="🟢",
            )

        st.markdown(
            "- If stress episodes are frequent across multiple sessions, "
            "consider discussing with a wellbeing advisor."
        )

    else:
        st.warning("No windows available for this subject.")


# ══════════════════════════════════════════════════════════════════════
# TAB 5: Evaluation (LOSO)
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Model Evaluation — Leave-One-Subject-Out")
    st.markdown("Binary classification: **Stress vs Non-Stress** evaluated across all subjects.")

    loso_results = run_loso_evaluation(all_features)

    # Overall metrics
    st.subheader("Overall Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LOSO Accuracy", f"{loso_results['overall_accuracy']:.1%}")
    with col2:
        st.metric("Weighted F1-Score", f"{loso_results['overall_f1_weighted']:.1%}")
    with col3:
        n_subjects = len(loso_results["per_subject"])
        st.metric("Subjects Evaluated", n_subjects)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        cm = loso_results["confusion_matrix"]
        target_names = loso_results["target_names"]

        fig = px.imshow(
            cm, x=target_names, y=target_names,
            text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=370, margin=dict(l=65, r=10, t=10, b=10))
        st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

    with col_right:
        st.subheader("Per-Subject Accuracy")
        per_sub_df = pd.DataFrame(loso_results["per_subject"])

        fig = px.bar(
            per_sub_df, x="subject", y="accuracy",
            color="accuracy", color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            labels={"subject": "Subject", "accuracy": "Accuracy"},
        )
        fig.add_hline(
            y=loso_results["overall_accuracy"],
            line_dash="dash", line_color="#aaaaaa", line_width=1.5,
            annotation_text=f"Mean: {loso_results['overall_accuracy']:.1%}",
            annotation_font=dict(size=12, color="#666666"),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=370, margin=dict(l=65, r=10, t=10, b=10))
        st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

    # Feature importance
    st.divider()
    st.subheader("Feature Importance — Top 15")
    st.markdown(
        "Which signal features contribute most to stress detection? "
        "This reveals which physiological channels are most informative."
    )

    importances = trained_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=True).tail(15)

    # Colour by signal source
    def get_signal_source(feat_name):
        if feat_name.startswith("EDA"):
            return "EDA"
        elif feat_name.startswith("BVP"):
            return "BVP"
        elif feat_name.startswith("TEMP"):
            return "Temperature"
        elif feat_name.startswith("ACC"):
            return "Accelerometer"
        return "Other"

    feat_imp_df["Signal"] = feat_imp_df["Feature"].apply(get_signal_source)

    signal_colors = {
        "EDA": "#f39c12",
        "BVP": "#e74c3c",
        "Temperature": "#3498db",
        "Accelerometer": "#2ecc71",
        "Other": "#95a5a6",
    }

    fig = px.bar(
        feat_imp_df, x="Importance", y="Feature",
        orientation="h", color="Signal",
        color_discrete_map=signal_colors,
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=450, margin=dict(l=65, r=10, t=10, b=10))
    st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

    # Signal-level importance summary
    signal_importance = {}
    for feat, imp in zip(feature_cols, importances):
        source = get_signal_source(feat)
        signal_importance[source] = signal_importance.get(source, 0) + imp

    sig_imp_df = pd.DataFrame(
        sorted(signal_importance.items(), key=lambda x: -x[1]),
        columns=["Signal Modality", "Total Importance"],
    )
    sig_imp_df["Contribution"] = (sig_imp_df["Total Importance"] / sig_imp_df["Total Importance"].sum() * 100).round(1)
    sig_imp_df["Contribution"] = sig_imp_df["Contribution"].astype(str) + "%"

    col_left2, col_right2 = st.columns([1, 2])
    with col_left2:
        st.markdown("**Signal-Level Importance:**")
        st.dataframe(sig_imp_df[["Signal Modality", "Contribution"]], use_container_width=True, hide_index=True)
    with col_right2:
        fig = px.pie(
            sig_imp_df, values="Total Importance", names="Signal Modality",
            color="Signal Modality", color_discrete_map=signal_colors,
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Importance by Signal Modality", font=dict(size=14)),
            height=280, margin=dict(l=65, r=10, t=40, b=10),
        )
        st.plotly_chart(apply_axis_spacing(fig), use_container_width=True)

    # Per-subject detail table
    st.divider()
    st.subheader("Per-Subject LOSO Results")
    per_sub_df_display = per_sub_df.copy()
    per_sub_df_display["accuracy"] = per_sub_df_display["accuracy"].apply(lambda x: f"{x:.1%}")
    per_sub_df_display["f1_weighted"] = per_sub_df_display["f1_weighted"].apply(lambda x: f"{x:.1%}")
    per_sub_df_display.columns = ["Subject", "Accuracy", "F1 (Weighted)", "Test Windows", "Stress Ratio"]
    per_sub_df_display["Stress Ratio"] = per_sub_df_display["Stress Ratio"].apply(
        lambda x: f"{x:.1%}" if x is not None else "N/A"
    )
    st.dataframe(per_sub_df_display, use_container_width=True)

    # Classification report
    st.subheader("Classification Report")
    report = loso_results["classification_report"]
    report_df = pd.DataFrame(report).T
    report_df = report_df.loc[
        [n for n in target_names] + ["weighted avg"],
        ["precision", "recall", "f1-score", "support"],
    ]
    report_df[["precision", "recall", "f1-score"]] = report_df[["precision", "recall", "f1-score"]].apply(
        lambda col: col.map(lambda x: f"{x:.3f}")
    )
    report_df["support"] = report_df["support"].astype(int)
    st.dataframe(report_df, use_container_width=True)

    # Critical Interpretation
    st.divider()
    st.subheader("Critical Interpretation")

    acc_std = np.std([s["accuracy"] for s in loso_results["per_subject"]])
    min_acc = min(s["accuracy"] for s in loso_results["per_subject"])
    max_acc = max(s["accuracy"] for s in loso_results["per_subject"])

    # Find most important signal
    top_signal = sig_imp_df.iloc[0]["Signal Modality"]

    st.markdown(f"""
**Key observations from LOSO evaluation:**

- Overall LOSO accuracy: **{loso_results['overall_accuracy']:.1%}**, weighted
  F1-score: **{loso_results['overall_f1_weighted']:.1%}** across {n_subjects} subjects.
- Per-subject accuracy ranges from **{min_acc:.1%}** to **{max_acc:.1%}**
  (std: {acc_std:.1%}), illustrating **individual variability** in stress detection.
- **{top_signal}** features contribute most to classification, consistent with
  literature on autonomic stress responses.
- This variance supports the report's central argument: strong aggregate
  benchmark accuracy does not guarantee reliable performance for every individual.
- LOSO ensures no data from the test subject is used during training,
  providing a more realistic estimate for unseen users.
    """)

    if not is_real_data:
        st.warning(
            "This evaluation uses WESAD-inspired synthetic data for demonstration. "
            "Results on real WESAD data may differ.",
            icon="⚠️",
        )
    else:
        st.success("This evaluation uses real WESAD benchmark data.", icon="✅")

    st.markdown("""
**Limitations:**

- The Random Forest with statistical features is a baseline approach.
  More advanced methods (e.g., deep learning) may improve performance.
- Real-world deployment faces additional challenges: motion artefacts
  during daily activities, inconsistent wearing, varying environments,
  and the absence of reliable ground-truth stress labels.
    """)
