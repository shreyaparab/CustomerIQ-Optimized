"""Persistence and batch prediction."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from configs.config import ARTIFACT_DIR, ID_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.data.validation import clean_dataframe, validate_features


def save_model(result, directory: Path | None = None) -> Path:
    directory = Path(directory or ARTIFACT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    version = result.metadata["model_version"]
    path = directory / f"model_{version}.joblib"
    joblib.dump({"pipeline": result.pipeline, "metadata": result.metadata}, path)
    meta = dict(result.metadata)
    meta["test_metrics"] = result.test_metrics
    (directory / f"model_{version}.json").write_text(json.dumps(meta, indent=2))
    (directory / "LATEST").write_text(version)
    return path


def load_model(directory: Path | None = None, version: str | None = None):
    directory = Path(directory or ARTIFACT_DIR)
    if version is None:
        latest = directory / "LATEST"
        if not latest.exists():
            raise FileNotFoundError("No trained model found. Train one in the Model Lab first.")
        version = latest.read_text().strip()
    path = directory / f"model_{version}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    blob = joblib.load(path)
    return blob["pipeline"], blob["metadata"]


def predict_batch(pipeline, metadata: dict, df: pd.DataFrame,
                  threshold: float | None = None) -> pd.DataFrame:
    """Return one score per input row; raises ValueError on invalid schema."""
    res = validate_features(df)
    if not res.ok:
        raise ValueError("; ".join(res.errors))
    clean = clean_dataframe(df)
    X = clean[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    proba = pipeline.predict_proba(X)[:, 1]
    if not np.all(np.isfinite(proba)):
        raise ValueError("Model produced non-finite scores.")
    thr = metadata["threshold"] if threshold is None else threshold
    out = pd.DataFrame(index=df.index)
    for c in ID_COLUMNS:
        if c in df.columns:
            out[c] = df[c]
    out["purchase_probability"] = proba
    out["predicted_label"] = (proba >= thr).astype(int)
    out["rank"] = out["purchase_probability"].rank(method="first", ascending=False).astype(int)
    out["model_version"] = metadata["model_version"]
    return out.sort_values("rank")
