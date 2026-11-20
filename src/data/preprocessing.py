"""Preprocessing: ColumnTransformer fitted only inside Pipelines (no leakage)."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.config import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED, TARGET,
)


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC_FEATURES),
            ("cat", Pipeline(cat_steps), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
    """Stratified split BEFORE any preprocessing is fitted."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
