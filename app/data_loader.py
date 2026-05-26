"""
data_loader.py
--------------
Loads and validates the Heart Disease UCI dataset.
Provides a clean DataFrame to the rest of the application.

Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
Features: 13 clinical features + 1 binary target (0 = no disease, 1 = disease)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent / "data" / "heart.csv"

# Human-readable feature descriptions for use in UI tooltips and SHAP plots
FEATURE_DESCRIPTIONS = {
    "age":      "Age of the patient (years)",
    "sex":      "Sex (1 = male, 0 = female)",
    "cp":       "Chest pain type (0–3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol":     "Serum cholesterol (mg/dl)",
    "fbs":      "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg":  "Resting ECG results (0–2)",
    "thalach":  "Maximum heart rate achieved",
    "exang":    "Exercise-induced angina (1 = yes)",
    "oldpeak":  "ST depression induced by exercise",
    "slope":    "Slope of peak exercise ST segment (0–2)",
    "ca":       "Number of major vessels coloured by fluoroscopy (0–3)",
    "thal":     "Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversible)",
}

TARGET_COL = "target"
FEATURE_COLS = list(FEATURE_DESCRIPTIONS.keys())


# ── Core loader ───────────────────────────────────────────────────────────────

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the heart disease CSV and run basic validation.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with 13 feature columns + 'target' column.

    Raises
    ------
    FileNotFoundError
        If heart.csv is not found at the expected path.
    ValueError
        If required columns are missing or target values are unexpected.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset\n"
            "and save as data/heart.csv"
        )

    df = pd.read_csv(path)

    # ── Validation ────────────────────────────────────────────────────────────
    expected_cols = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1.")

    # Drop any duplicate rows silently
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Split DataFrame into feature matrix X and target vector y.

    Returns
    -------
    X : pd.DataFrame  — shape (n_samples, 13)
    y : pd.Series     — binary target (0/1)
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics with a 'disease_rate' row appended.
    Used in the Dataset Explorer panel of the dashboard.
    """
    stats = df.describe().T  # transpose: features as rows
    stats["disease_rate"] = df.groupby(TARGET_COL).size().get(1, 0) / len(df)
    return stats.round(3)