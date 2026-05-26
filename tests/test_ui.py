"""
test_ui.py
----------
End-to-end smoke tests that mirror what the UI does at startup.
Verifies the full pipeline: data → train → explain → visualise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.data_loader import load_data, FEATURE_COLS, FEATURE_DESCRIPTIONS
from app.trainer import train_and_evaluate, load_model, load_scaler
from app.explainer import get_explainer, get_global_shap_values, explain_single
from app.visualiser import (
    plot_global_importance,
    plot_shap_beeswarm,
    plot_waterfall,
    plot_model_comparison,
    plot_feature_distribution,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    """Full pipeline output — trained once, shared across all tests."""
    output     = train_and_evaluate()
    explainer  = get_explainer()
    global_shap = get_global_shap_values(output["X_test"], explainer)
    return {
        "output":       output,
        "explainer":    explainer,
        "global_shap":  global_shap,
        "scaler":       load_scaler(),
        "df":           load_data(),
    }


# ── Data layer ────────────────────────────────────────────────────────────────

def test_dataset_loads(pipeline):
    assert len(pipeline["df"]) > 200


def test_feature_descriptions_complete():
    for col in FEATURE_COLS:
        assert col in FEATURE_DESCRIPTIONS
        assert len(FEATURE_DESCRIPTIONS[col]) > 5


# ── Training layer ────────────────────────────────────────────────────────────

def test_both_models_in_results(pipeline):
    results = pipeline["output"]["results"]
    assert "Random Forest" in results
    assert "Logistic Regression" in results


def test_test_split_non_empty(pipeline):
    assert len(pipeline["output"]["X_test"]) > 0
    assert len(pipeline["output"]["y_test"]) > 0


def test_scaler_loaded(pipeline):
    assert pipeline["scaler"] is not None


# ── Explainer layer ───────────────────────────────────────────────────────────

def test_global_shap_keys(pipeline):
    for key in ["shap_values", "mean_abs_shap", "feature_names", "X"]:
        assert key in pipeline["global_shap"]


def test_explain_single_runs(pipeline):
    X_test = pipeline["output"]["X_test"]
    result = explain_single(X_test.iloc[[0]], pipeline["explainer"])
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in {0, 1}


def test_what_if_pipeline(pipeline):
    """Simulate what the What-If panel does: scale raw input → explain."""
    df      = pipeline["df"]
    scaler  = pipeline["scaler"]

    # Build a mean-value patient (what the sliders default to)
    user_input = {f: float(df[f].mean()) for f in FEATURE_COLS}
    input_df   = pd.DataFrame([user_input])
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=FEATURE_COLS,
    )

    result = explain_single(input_scaled, pipeline["explainer"])
    assert result["shap_values"].shape == (len(FEATURE_COLS),)


# ── Visualiser layer ──────────────────────────────────────────────────────────

def test_all_charts_return_figures(pipeline):
    g        = pipeline["global_shap"]
    results  = pipeline["output"]["results"]
    X_test   = pipeline["output"]["X_test"]
    df       = pipeline["df"]
    explainer = pipeline["explainer"]

    result = explain_single(X_test.iloc[[0]], explainer)

    charts = [
        plot_global_importance(g["mean_abs_shap"]),
        plot_shap_beeswarm(g["shap_values"], g["X"]),
        plot_waterfall(result["shap_values"], result["base_value"],
                       FEATURE_COLS, X_test.iloc[0]),
        plot_model_comparison(results),
        plot_feature_distribution(df, "age"),
    ]

    for fig in charts:
        assert isinstance(fig, go.Figure)