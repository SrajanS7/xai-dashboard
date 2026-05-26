"""
explainer.py
------------
Computes SHAP values for the trained Random Forest classifier.

Provides:
  - Global feature importance (mean |SHAP| across all samples)
  - Per-sample SHAP values for individual prediction explanations
  - Force plot data for the what-if simulator

SHAP (SHapley Additive exPlanations) assigns each feature a contribution
value for a specific prediction — grounded in cooperative game theory.
"""

import numpy as np
import pandas as pd
import shap
from app.trainer import load_model, load_scaler, train_and_evaluate
from app.data_loader import load_data, get_features_and_target, FEATURE_COLS

# ── Explainer builder ─────────────────────────────────────────────────────────

def get_explainer(model=None):
    """
    Build and return a SHAP TreeExplainer for the Random Forest.

    TreeExplainer is exact (not approximate) for tree-based models —
    much faster than KernelExplainer and fully deterministic.

    Parameters
    ----------
    model : trained sklearn model, optional
        If None, loads from disk.
    """
    if model is None:
        model = load_model("Random Forest")
    explainer = shap.TreeExplainer(model)
    return explainer


# ── Global explanations ───────────────────────────────────────────────────────

def get_global_shap_values(X: pd.DataFrame, explainer=None) -> dict:
    """
    Compute SHAP values for all samples and return global importance.

    Parameters
    ----------
    X : pd.DataFrame — feature matrix (scaled)
    explainer : shap.TreeExplainer, optional

    Returns
    -------
    dict with keys:
      - 'shap_values'     : np.ndarray shape (n_samples, n_features)
      - 'mean_abs_shap'   : pd.Series — mean |SHAP| per feature, sorted desc
      - 'feature_names'   : list of feature names
      - 'X'               : original feature DataFrame
    """
    if explainer is None:
        explainer = get_explainer()

    # shap_values shape: (n_samples, n_features) for binary class 1
    shap_values = explainer.shap_values(X)

    # For RandomForest binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    return {
        "shap_values":   shap_values,
        "mean_abs_shap": mean_abs_shap,
        "feature_names": FEATURE_COLS,
        "X":             X,
    }


# ── Individual explanations ───────────────────────────────────────────────────
def explain_single(row: pd.DataFrame, explainer=None) -> dict:
    """
    Compute SHAP values for a single patient row.

    Parameters
    ----------
    row : pd.DataFrame — single row, shape (1, 13), already scaled

    Returns
    -------
    dict with keys:
      - 'shap_values'   : np.ndarray shape (13,)
      - 'base_value'    : float — expected model output (baseline)
      - 'prediction'    : int   — 0 or 1
      - 'probability'   : float — P(heart disease)
      - 'top_features'  : pd.DataFrame — top 5 contributors sorted by |SHAP|
    """
    if explainer is None:
        explainer = get_explainer()

    model = load_model("Random Forest")

    shap_values = explainer.shap_values(row)

    # Handle both list format and 3D array format depending on SHAP version
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_flat = shap_values[0]  # shape (13,)

    prediction  = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    # Handle expected_value for both list and array formats
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        base_value = float(expected_value[1])
    else:
        base_value = float(expected_value)

    # Top 5 features by absolute SHAP value
    top_features = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "shap_value": shap_flat,
        "abs_shap":   np.abs(shap_flat),
    }).sort_values("abs_shap", ascending=False).head(5)

    return {
        "shap_values":  shap_flat,
        "base_value":   base_value,
        "prediction":   prediction,
        "probability":  probability,
        "top_features": top_features,
    }

# ── Convenience loader ────────────────────────────────────────────────────────

def load_test_data() -> tuple:
    """
    Load and return scaled test split — used by UI and tests.

    Returns
    -------
    X_test : pd.DataFrame
    y_test : pd.Series
    """
    output = train_and_evaluate()
    return output["X_test"], output["y_test"]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Computing SHAP values...")
    X_test, y_test = load_test_data()
    explainer      = get_explainer()
    global_shap    = get_global_shap_values(X_test, explainer)

    print("\nTop 5 most important features (mean |SHAP|):")
    print(global_shap["mean_abs_shap"].head())

    print("\nExplaining first test sample...")
    single = explain_single(X_test.iloc[[0]], explainer)
    print(f"  Prediction : {'Disease' if single['prediction'] else 'No Disease'}")
    print(f"  Probability: {single['probability']:.3f}")
    print(f"  Top features:\n{single['top_features'][['feature','shap_value']].to_string(index=False)}")