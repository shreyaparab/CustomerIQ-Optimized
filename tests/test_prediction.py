import numpy as np
import pytest

from src.models.predict import load_model, predict_batch, save_model


def test_comparison_includes_baseline(trained):
    assert "Dummy (baseline)" in set(trained.comparison["model"])
    assert trained.test_metrics["roc_auc"] > 0.5


def test_one_valid_score_per_row(trained, df):
    inp = df.drop(columns=["ProdTaken"]).head(120)
    out = predict_batch(trained.pipeline, trained.metadata, inp)
    assert len(out) == len(inp)
    p = out["purchase_probability"].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert set(out["predicted_label"].unique()) <= {0, 1}
    assert sorted(out["rank"]) == list(range(1, len(inp) + 1))


def test_missing_column_raises_clear_error(trained, df):
    with pytest.raises(ValueError, match="Missing required columns"):
        predict_batch(trained.pipeline, trained.metadata, df.drop(columns=["Passport"]).head(5))


def test_model_roundtrip_after_restart(trained, df, tmp_path):
    save_model(trained, tmp_path)
    pipe, meta = load_model(tmp_path)
    inp = df.drop(columns=["ProdTaken"]).head(40)
    a = predict_batch(trained.pipeline, trained.metadata, inp)["purchase_probability"].sort_index()
    b = predict_batch(pipe, meta, inp)["purchase_probability"].sort_index()
    assert np.allclose(a, b)
    assert meta["model_version"] == trained.metadata["model_version"]


def test_load_without_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train one"):
        load_model(tmp_path)


def test_target_never_a_feature(trained):
    assert "ProdTaken" not in trained.metadata["numeric_features"]
    assert "ProdTaken" not in trained.metadata["categorical_features"]
    assert "CustomerID" not in trained.metadata["numeric_features"]
