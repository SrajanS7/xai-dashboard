"""
trainer.py
----------
Trains and evaluates two classifiers on the Heart Disease UCI dataset:
  - Random Forest   → better accuracy, richer SHAP explanations
  - Logistic Regression → fast, interpretable baseline

Models are serialised to models/ and reloaded by the SHAP explainer.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    f1_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from app.data_loader import load_data, get_features_and_target

# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RF_PATH  = MODELS_DIR / "random_forest.pkl"
LR_PATH  = MODELS_DIR / "logistic_reg.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# ── Model definitions ─────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Return untrained model instances.
    Centralised here so hyperparameters are easy to tune later.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        ),
    }


# ── Training pipeline ─────────────────────────────────────────────────────────

def train_and_evaluate(test_size: float = 0.2) -> dict:
    """
    Full training pipeline:
      1. Load + split data
      2. Scale features (needed for LR; doesn't hurt RF)
      3. Train both models
      4. Evaluate on hold-out test set
      5. Serialise models + scaler to disk

    Returns
    -------
    dict with keys: 'results', 'X_test', 'y_test', 'X_train', 'y_train'
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_data()
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ── Scale features ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Keep as DataFrames for SHAP (preserves feature names)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X.columns)

    joblib.dump(scaler, SCALER_PATH)

    # ── Train + evaluate ──────────────────────────────────────────────────────
    models  = build_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # 5-fold CV on full training set for robustness
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="roc_auc"
        )

        results[name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "cv_auc_std":  round(cv_scores.std(), 4),
            "report":    classification_report(y_test, y_pred),
        }

    # ── Serialise ─────────────────────────────────────────────────────────────
    joblib.dump(models["Random Forest"],      RF_PATH)
    joblib.dump(models["Logistic Regression"], LR_PATH)

    print_comparison(results)

    return {
        "results":  results,
        "X_test":   X_test_scaled,
        "y_test":   y_test,
        "X_train":  X_train_scaled,
        "y_train":  y_train,
    }


# ── Loaders (used by explainer + UI) ─────────────────────────────────────────

def load_model(name: str):
    """
    Load a serialised model by name.

    Parameters
    ----------
    name : "Random Forest" | "Logistic Regression"
    """
    path = RF_PATH if name == "Random Forest" else LR_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_and_evaluate() first."
        )
    return joblib.load(path)


def load_scaler():
    """Load the fitted StandardScaler."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError("Scaler not found. Run train_and_evaluate() first.")
    return joblib.load(SCALER_PATH)


# ── Utilities ─────────────────────────────────────────────────────────────────

def print_comparison(results: dict) -> None:
    """Pretty-print model comparison to terminal."""
    print("\n" + "="*55)
    print(f"{'Model':<25} {'Acc':>6} {'AUC':>6} {'F1':>6} {'CV-AUC':>8}")
    print("="*55)
    for name, m in results.items():
        print(
            f"{name:<25} {m['accuracy']:>6} {m['roc_auc']:>6} "
            f"{m['f1']:>6} {m['cv_auc_mean']:>6}±{m['cv_auc_std']}"
        )
    print("="*55 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_evaluate()