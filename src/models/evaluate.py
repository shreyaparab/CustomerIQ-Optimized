"""Evaluation helpers: metrics, threshold choice, calibration."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def pick_threshold(y_true, proba, metric: str = "f1") -> float:
    """Choose a decision threshold on validation/OOF data (not on test)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    candidates = np.unique(np.round(proba, 4))
    if len(candidates) > 200:
        candidates = np.quantile(proba, np.linspace(0.01, 0.99, 99))
    best_t, best_s = 0.5, -1.0
    for t in candidates:
        pred = (proba >= t).astype(int)
        s = f1_score(y_true, pred, zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return float(best_t)


def compute_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "support_pos": int((y_true == 1).sum()),
        "support_neg": int((y_true == 0).sum()),
        "brier": float(brier_score_loss(y_true, proba)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return out


def evaluate_on_split(pipeline, X, y, threshold: float) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    m = compute_metrics(y, proba, threshold)
    m["calibration"] = calibration_table(y, proba)
    return m


def calibration_table(y_true, proba, n_bins: int = 8) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, proba, n_bins=n_bins, strategy="quantile")
        return {"mean_predicted": mean_pred.tolist(),
                "fraction_positive": frac_pos.tolist()}
    except ValueError:
        return {"mean_predicted": [], "fraction_positive": []}
