"""Smoke tests for visualiser.py"""

import pytest
import plotly.graph_objects as go
from app.explainer import get_explainer, get_global_shap_values, explain_single, load_test_data
from app.trainer import train_and_evaluate
from app.data_loader import load_data, FEATURE_COLS
from app.visualiser import (
    plot_global_importance,
    plot_shap_beeswarm,
    plot_waterfall,
    plot_model_comparison,
    plot_feature_distribution,
)


@pytest.fixture(scope="module")
def test_data():
    return load_test_data()


@pytest.fixture(scope="module")
def explainer():
    return get_explainer()


@pytest.fixture(scope="module")
def global_shap(test_data, explainer):
    X_test, _ = test_data
    return get_global_shap_values(X_test, explainer)


@pytest.fixture(scope="module")
def training_results():
    return train_and_evaluate()["results"]


def test_global_importance_returns_figure(global_shap):
    fig = plot_global_importance(global_shap["mean_abs_shap"])
    assert isinstance(fig, go.Figure)


def test_beeswarm_returns_figure(global_shap):
    fig = plot_shap_beeswarm(global_shap["shap_values"], global_shap["X"])
    assert isinstance(fig, go.Figure)


def test_waterfall_returns_figure(test_data, explainer):
    X_test, _ = test_data
    result = explain_single(X_test.iloc[[0]], explainer)
    fig = plot_waterfall(
        result["shap_values"],
        result["base_value"],
        FEATURE_COLS,
        X_test.iloc[0],
    )
    assert isinstance(fig, go.Figure)


def test_model_comparison_returns_figure(training_results):
    fig = plot_model_comparison(training_results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # accuracy, auc, f1


def test_feature_distribution_returns_figure():
    df = load_data()
    fig = plot_feature_distribution(df, "age")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # one trace per class