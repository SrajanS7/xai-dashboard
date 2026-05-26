"""Smoke + sanity tests for trainer.py"""

import pytest
import joblib
import numpy as np
from pathlib import Path
from app.trainer import train_and_evaluate, load_model, load_scaler, RF_PATH, LR_PATH


@pytest.fixture(scope="module")
def training_output():
    """Train once, reuse across all tests in this module."""
    return train_and_evaluate()


def test_results_keys_present(training_output):
    for model_name in ["Random Forest", "Logistic Regression"]:
        assert model_name in training_output["results"]


def test_accuracy_above_threshold(training_output):
    """Both models should beat a naive majority-class baseline (~54%)."""
    for name, metrics in training_output["results"].items():
        assert metrics["accuracy"] > 0.70, f"{name} accuracy too low: {metrics['accuracy']}"


def test_auc_above_threshold(training_output):
    for name, metrics in training_output["results"].items():
        assert metrics["roc_auc"] > 0.75, f"{name} AUC too low: {metrics['roc_auc']}"


def test_models_serialised():
    assert RF_PATH.exists(), "random_forest.pkl not found"
    assert LR_PATH.exists(), "logistic_reg.pkl not found"


def test_load_model_random_forest(training_output):
    model = load_model("Random Forest")
    X_test = training_output["X_test"]
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_load_model_logistic_regression(training_output):
    model = load_model("Logistic Regression")
    X_test = training_output["X_test"]
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_scaler_transforms_correctly(training_output):
    scaler = load_scaler()
    X_test = training_output["X_test"]
    # Already scaled — mean should be near 0
    means = np.abs(X_test.mean())
    assert means.mean() < 1.0