"""K-Means segmentation with silhouette selection, PCA projection and profiles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from configs.config import CLUSTER_FEATURES, SEED, TARGET


def _prep() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])


def evaluate_k_range(df: pd.DataFrame, k_min: int = 2, k_max: int = 8,
                     seed: int = SEED) -> pd.DataFrame:
    """Inertia and silhouette for a range of K (aids, not automatic truth)."""
    feats = [c for c in CLUSTER_FEATURES if c in df.columns]
    Z = _prep().fit_transform(df[feats])
    k_max = min(k_max, len(df) - 1)
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": float(silhouette_score(Z, km.labels_))})
    return pd.DataFrame(rows)


@dataclass
class SegmentationResult:
    labels: np.ndarray
    pca_coords: np.ndarray
    explained_variance: list[float]
    profile: pd.DataFrame
    silhouette: float
    k: int
    features: list[str]


def fit_segments(df: pd.DataFrame, k: int, seed: int = SEED) -> SegmentationResult:
    feats = [c for c in CLUSTER_FEATURES if c in df.columns]
    if not feats:
        raise ValueError("None of the clustering features are present.")
    if k < 2 or k >= len(df):
        raise ValueError("k must be >= 2 and smaller than the number of rows.")
    prep = _prep()
    Z = prep.fit_transform(df[feats])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    sil = float(silhouette_score(Z, km.labels_))
    pca = PCA(n_components=2, random_state=seed).fit(Z)
    coords = pca.transform(Z)

    tmp = df[feats].copy()
    tmp["segment"] = km.labels_
    profile = tmp.groupby("segment").mean(numeric_only=True)
    profile.insert(0, "size", tmp.groupby("segment").size())
    profile.insert(1, "share", profile["size"] / len(tmp))
    if TARGET in df.columns:
        # Descriptive association only, not causal.
        profile["observed_purchase_rate"] = (
            df.assign(segment=km.labels_).groupby("segment")[TARGET].mean()
        )
    return SegmentationResult(km.labels_, coords,
                              pca.explained_variance_ratio_.tolist(),
                              profile.round(3), sil, k, feats)


def suggest_label(profile_row: pd.Series, overall: pd.DataFrame) -> str:
    """Draft human-reviewable name from the most distinctive feature."""
    feats = [c for c in overall.columns if c not in ("size", "share", "observed_purchase_rate")]
    z = {}
    for f in feats:
        sd = overall[f].std()
        if sd and not np.isnan(sd):
            z[f] = (profile_row[f] - overall[f].mean()) / sd
    if not z:
        return "Segment"
    top = max(z, key=lambda k: abs(z[k]))
    return f"{'High' if z[top] > 0 else 'Low'} {top}"
