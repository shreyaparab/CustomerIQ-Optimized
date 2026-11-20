import numpy as np

from configs.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.data.preprocessing import build_preprocessor, split_data


def test_split_is_stratified_and_disjoint(df):
    Xtr, Xte, ytr, yte = split_data(df)
    assert set(Xtr.index).isdisjoint(Xte.index)
    assert abs(ytr.mean() - yte.mean()) < 0.03


def test_preprocessor_fitted_on_train_only(df):
    Xtr, Xte, _, _ = split_data(df)
    prep = build_preprocessor().fit(Xtr)
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    # Learned mean must equal the TRAIN-set statistic, not the full-data one.
    imputed = prep.named_transformers_["num"].named_steps["impute"].transform(Xtr[NUMERIC_FEATURES])
    assert np.allclose(scaler.mean_, imputed.mean(axis=0))


def test_unseen_category_does_not_crash(df):
    Xtr, Xte, _, _ = split_data(df)
    prep = build_preprocessor().fit(Xtr)
    row = Xte.iloc[[0]].copy()
    row["Occupation"] = "Astronaut"
    out = prep.transform(row)
    assert np.isfinite(out).all()


def test_no_missing_after_transform(df):
    Xtr, _, _, _ = split_data(df)
    out = build_preprocessor().fit_transform(Xtr)
    assert not np.isnan(out).any()
