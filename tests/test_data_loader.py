"""Basic smoke tests for data_loader.py"""

import pytest
import pandas as pd
from app.data_loader import load_data, get_features_and_target, FEATURE_COLS, TARGET_COL


def test_load_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)


def test_expected_columns_present():
    df = load_data()
    for col in FEATURE_COLS + [TARGET_COL]:
        assert col in df.columns, f"Missing column: {col}"


def test_target_is_binary():
    df = load_data()
    assert set(df[TARGET_COL].unique()).issubset({0, 1})


def test_no_nulls():
    df = load_data()
    assert df.isnull().sum().sum() == 0


def test_feature_target_split():
    df = load_data()
    X, y = get_features_and_target(df)
    assert X.shape[1] == len(FEATURE_COLS)
    assert len(y) == len(X)