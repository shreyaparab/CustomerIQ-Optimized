"""Explainability: SHAP when available, permutation importance as fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from configs.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED

try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    shap = None
    HAS_SHAP = False

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    prep = pipeline.named_steps["prep"]
    Z = prep.transform(X)
    return pd.DataFrame(Z, columns=prep.get_feature_names_out(), index=X.index)


def global_importance(pipeline, X: pd.DataFrame, y: pd.Series,
                      max_rows: int = 500) -> tuple[pd.DataFrame, str]:
    """Return (importance table, method name). Never raises on SHAP errors."""
    if len(X) > max_rows:
        idx = X.sample(max_rows, random_state=SEED).index
        X, y = X.loc[idx], y.loc[idx]
    if HAS_SHAP:
        try:
            clf = pipeline.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                Z = transformed_frame(pipeline, X)
                sv = shap.TreeExplainer(clf).shap_values(Z)
                sv = _positive_class(sv)
                imp = pd.DataFrame({
                    "feature": Z.columns,
                    "importance": np.abs(sv).mean(axis=0),
                }).sort_values("importance", ascending=False)
                return imp.reset_index(drop=True), "SHAP (mean |value|)"
        except Exception:
            pass
    r = permutation_importance(pipeline, X, y, n_repeats=5, random_state=SEED,
                               scoring="average_precision", n_jobs=1)
    imp = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean}
                       ).sort_values("importance", ascending=False)
    return imp.reset_index(drop=True), "Permutation importance (PR-AUC drop)"


def _positive_class(sv) -> np.ndarray:
    if isinstance(sv, list):
        return np.asarray(sv[-1])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, -1]
    return sv


def local_explanation(pipeline, X_row: pd.DataFrame, X_background: pd.DataFrame,
                      top_n: int = 10) -> tuple[pd.DataFrame, str]:
    """Per-customer contributions. SHAP for tree models, else a one-at-a-time
    substitution approach (model-agnostic, approximate)."""
    if HAS_SHAP:
        try:
            clf = pipeline.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                Z = transformed_frame(pipeline, X_row)
                sv = _positive_class(shap.TreeExplainer(clf).shap_values(Z))[0]
                df = pd.DataFrame({"feature": Z.columns, "contribution": sv,
                                   "value": Z.iloc[0].values})
                df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
                return df.head(top_n).reset_index(drop=True), "SHAP"
        except Exception:
            pass
    base = float(pipeline.predict_proba(X_row)[0, 1])
    rows = []
    for col in X_row.columns:
        probs = []
        for v in X_background[col].dropna().sample(
                min(30, X_background[col].notna().sum()), random_state=SEED):
            tmp = X_row.copy()
            tmp[col] = v
            probs.append(pipeline.predict_proba(tmp)[0, 1])
        rows.append({"feature": col, "contribution": base - float(np.mean(probs)),
                     "value": X_row[col].iloc[0]})
    df = pd.DataFrame(rows)
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
    return df.head(top_n).reset_index(drop=True), "Substitution (approximate)"
