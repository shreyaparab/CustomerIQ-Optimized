"""Simple drift checks: PSI for numeric columns, share shift for categoricals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from configs.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = pd.to_numeric(expected, errors="coerce").dropna()
    a = pd.to_numeric(actual, errors="coerce").dropna()
    if len(e) < 2 or len(a) < 2 or e.nunique() < 2:
        return 0.0
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUMERIC_FEATURES:
        if c in reference and c in current:
            v = psi(reference[c], current[c])
            rows.append({"feature": c, "type": "numeric", "score": v,
                         "status": "high" if v > 0.25 else "moderate" if v > 0.1 else "ok"})
    for c in CATEGORICAL_FEATURES:
        if c in reference and c in current:
            r = reference[c].value_counts(normalize=True)
            a = current[c].value_counts(normalize=True)
            cats = r.index.union(a.index)
            v = float(0.5 * np.abs(r.reindex(cats, fill_value=0)
                                   - a.reindex(cats, fill_value=0)).sum())
            rows.append({"feature": c, "type": "categorical", "score": v,
                         "status": "high" if v > 0.2 else "moderate" if v > 0.1 else "ok"})
    return pd.DataFrame(rows)
