"""Model training and comparison with cross-validation and a held-out test set."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from configs.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED
from src.data.preprocessing import build_preprocessor, split_data
from src.models.evaluate import evaluate_on_split, pick_threshold

try:  # optional dependency
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    HAS_XGB = False


def candidate_models(pos_weight: float = 1.0, seed: int = SEED) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "Dummy (baseline)": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", DummyClassifier(strategy="prior")),
        ]),
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "Decision Tree": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                                           class_weight="balanced", random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=1,
                                           n_jobs=-1, random_state=seed)),
        ]),
        "Extra Trees": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", ExtraTreesClassifier(n_estimators=1000, n_jobs=-1,
                                         random_state=seed)),
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
                random_state=seed)),
        ]),
        "Ensemble (ET + HGB + RF)": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", VotingClassifier(estimators=[
                ("et", ExtraTreesClassifier(n_estimators=600, n_jobs=-1, random_state=seed)),
                ("hgb", HistGradientBoostingClassifier(
                    max_iter=400, learning_rate=0.1, max_leaf_nodes=63, random_state=seed)),
                ("rf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed)),
            ], voting="soft")),
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                scale_pos_weight=pos_weight, random_state=seed, n_jobs=1,
            )),
        ])
    return models


@dataclass
class TrainingResult:
    best_name: str
    pipeline: Pipeline
    threshold: float
    comparison: pd.DataFrame
    test_metrics: dict
    metadata: dict = field(default_factory=dict)


def train_and_compare(df: pd.DataFrame, seed: int = SEED, cv_folds: int = 5,
                      selection_metric: str = "pr_auc") -> TrainingResult:
    """Compare models by CV on the training split, then evaluate once on test."""
    X_train, X_test, y_train, y_test = split_data(df, seed=seed)
    pos = max(int(y_train.sum()), 1)
    pos_weight = float((len(y_train) - pos) / pos)
    n_splits = max(2, min(cv_folds, int(y_train.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    for name, pipe in candidate_models(pos_weight, seed).items():
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, n_jobs=1,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision",
                     "accuracy": "accuracy",
                     "f1": make_scorer(f1_score, zero_division=0),
                     "precision": make_scorer(precision_score, zero_division=0),
                     "recall": make_scorer(recall_score, zero_division=0)},
        )
        row = {"model": name}
        for m in ("roc_auc", "pr_auc", "accuracy", "f1", "precision", "recall"):
            row[f"{m}_mean"] = float(np.mean(scores[f"test_{m}"]))
            row[f"{m}_std"] = float(np.std(scores[f"test_{m}"]))
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(
        f"{selection_metric}_mean", ascending=False
    ).reset_index(drop=True)

    best_name = comparison.iloc[0]["model"]
    best_pipe = candidate_models(pos_weight, seed)[best_name]

    # Threshold chosen on out-of-fold training predictions, never on test.
    from sklearn.model_selection import cross_val_predict
    if best_name.startswith("Dummy"):
        oof = np.full(len(y_train), float(y_train.mean()))
    else:
        oof = cross_val_predict(best_pipe, X_train, y_train, cv=cv,
                                method="predict_proba")[:, 1]
    threshold = pick_threshold(y_train.to_numpy(), oof, metric="f1")

    best_pipe.fit(X_train, y_train)
    test_metrics = evaluate_on_split(best_pipe, X_test, y_test, threshold)

    version = dt.datetime.now(dt.timezone.utc).strftime("v%Y%m%d%H%M%S")
    metadata = {
        "model_version": version,
        "model_name": best_name,
        "threshold": threshold,
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_mapping": {"0": "no purchase", "1": "purchase"},
        "selection_metric": selection_metric,
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return TrainingResult(best_name, best_pipe, threshold, comparison,
                          test_metrics, metadata)
