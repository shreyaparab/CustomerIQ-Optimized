import numpy as np
import pandas as pd

from src.data.validation import (
    clean_dataframe, quality_report, validate_features, validate_training,
)


def test_valid_frame_passes(df):
    assert validate_training(df).ok


def test_missing_column_message(df):
    res = validate_features(df.drop(columns=["Age"]))
    assert not res.ok
    assert "Age" in res.errors[0]


def test_empty_frame_rejected(df):
    assert not validate_features(df.iloc[0:0]).ok


def test_bad_target_labels_rejected(df):
    bad = df.copy()
    bad["ProdTaken"] = bad["ProdTaken"].replace({1: 2})
    res = validate_training(bad)
    assert not res.ok and any("0/1" in e for e in res.errors)


def test_missing_target_rejected(df):
    assert not validate_training(df.drop(columns=["ProdTaken"])).ok


def test_single_class_rejected(df):
    one = df.copy()
    one["ProdTaken"] = 0
    assert not validate_training(one).ok


def test_non_numeric_value_flagged(df):
    bad = df.copy().astype({"Age": object})
    bad.loc[bad.index[0], "Age"] = "abc"
    assert not validate_features(bad).ok


def test_negative_income_flagged(df):
    bad = df.copy()
    bad.loc[bad.index[0], "MonthlyIncome"] = -5
    assert not validate_features(bad).ok


def test_clean_fixes_known_typos():
    raw = pd.DataFrame({"Gender": ["Fe Male", " Male "], "MaritalStatus": ["Unmarried", "Married"]})
    out = clean_dataframe(raw)
    assert list(out["Gender"]) == ["Female", "Male"]
    assert out["MaritalStatus"].iloc[0] == "Single"


def test_quality_report(df):
    rep = quality_report(df)
    assert rep["rows"] == len(df)
    assert 0 < rep["positive_rate"] < 1
