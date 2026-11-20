import numpy as np
import pytest

from configs.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.explainability.shap_reports import global_importance, local_explanation
from src.monitoring.drift import drift_report, psi
from src.segmentation.cluster import evaluate_k_range, fit_segments
from src.storage.repository import list_events, list_prediction_runs, log_event, log_prediction_run


def test_segmentation_outputs(df):
    seg = fit_segments(df, 3)
    assert len(seg.labels) == len(df)
    assert set(seg.labels) == {0, 1, 2}
    assert seg.profile["size"].sum() == len(df)
    assert -1 <= seg.silhouette <= 1
    assert seg.pca_coords.shape == (len(df), 2)


def test_segmentation_excludes_target_and_id(df):
    seg = fit_segments(df, 3)
    assert "ProdTaken" not in seg.features and "CustomerID" not in seg.features


def test_invalid_k_rejected(df):
    with pytest.raises(ValueError):
        fit_segments(df, 1)


def test_k_range(df):
    assert len(evaluate_k_range(df, 2, 5)) == 4


def test_global_and_local_explanations(trained, df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    imp, method = global_importance(trained.pipeline, X, df["ProdTaken"], max_rows=200)
    assert len(imp) > 0 and method
    loc, lm = local_explanation(trained.pipeline, X.head(1), X)
    assert len(loc) > 0 and np.isfinite(loc["contribution"]).all()


def test_psi_identical_is_zero(df):
    assert psi(df["Age"], df["Age"]) == pytest.approx(0, abs=1e-6)


def test_drift_detects_shift(df):
    shifted = df.copy()
    shifted["MonthlyIncome"] = shifted["MonthlyIncome"] * 2
    rep = drift_report(df, shifted).set_index("feature")
    assert rep.loc["MonthlyIncome", "status"] == "high"


def test_repository_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    log_prediction_run("v1", 10, 0.3, 0.2, ["a", "b"], db_path=db)
    log_event("x", "y", db_path=db)
    assert list_prediction_runs(db_path=db)[0]["n_rows"] == 10
    assert list_events(db_path=db)[0]["event"] == "x"
