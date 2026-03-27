# StressLens — Wearable Stress Monitoring Prototype

A WESAD-guided proof-of-concept prototype for wrist-based wearable stress monitoring,
developed as part of COMP8230 (Mining Unstructured Data) at Macquarie University.

**Author:** Itsara Timaroon (48572918)
**School of Computing, Macquarie University — Semester 1, 2026**

> **Disclaimer:** This is an academic prototype for monitoring and decision-support
> demonstration only. It is NOT a clinical diagnostic tool.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### 3. (Optional) Using real WESAD data

By default, the prototype generates WESAD-inspired synthetic data for demonstration.
To use real WESAD benchmark data:

1. Download the WESAD dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
2. Extract and place subject folders in `wesad_data/`:
   ```
   wesad_data/
   ├── S2/S2.pkl
   ├── S3/S3.pkl
   ├── ...
   └── S17/S17.pkl
   ```
3. Re-run the app — it will automatically detect and use the real data.

## Project Structure

```
├── app.py                  # Streamlit dashboard (entry point)
├── data_generator.py       # WESAD loader + synthetic data fallback (30 subjects)
├── preprocessing.py        # Signal filtering, normalisation, windowing
├── feature_extraction.py   # Statistical feature extraction per window
├── model.py                # Random Forest + LOSO cross-validation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## What This Prototype Demonstrates

| Tab | Purpose |
|-----|---------|
| Signal Viewer | Visualise raw wrist sensor signals (BVP, EDA, TEMP, ACC) by subject |
| Preprocessing | Show filtering, normalisation, and window segmentation pipeline |
| Stress Inference | Per-window stress likelihood predictions with confidence scores |
| Trends & Episodes | Stress episode detection, timeline, and self-management suggestions |
| Evaluation (LOSO) | Leave-One-Subject-Out cross-validation with accuracy, F1, confusion matrix |

## Technical Details

- **Signals:** Blood Volume Pulse, Electrodermal Activity, Skin Temperature, 3-axis Accelerometer
- **Preprocessing:** Butterworth lowpass filtering, Z-score normalisation, 60s sliding windows (50% overlap)
- **Features:** 9 statistical features per signal per window (mean, std, min, max, range, median, IQR, slope, zero-crossings)
- **Classifier:** Random Forest (100 trees, balanced class weights, max depth 10)
- **Evaluation:** Leave-One-Subject-Out cross-validation, binary (Stress vs Non-Stress)

## What Is Real vs. Simulated

| Component | Status |
|-----------|--------|
| Preprocessing pipeline | Real (functional code) |
| Feature extraction | Real (functional code) |
| Random Forest classifier | Real (trained and evaluated) |
| LOSO cross-validation | Real (functional evaluation) |
| Wearable data (default) | Synthetic — mimics WESAD statistical properties |
| Wearable data (optional) | Real — if WESAD pickle files are provided |
| Stress episode detection | Simplified heuristic (consecutive window grouping) |
| Self-management suggestions | Rule-based demonstration text |

## Dependencies

- Python 3.10+
- streamlit >= 1.30.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- plotly >= 5.18.0
