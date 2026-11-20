"""Data validation: schema checks, label checks, and a data-quality report."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from configs.config import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, REQUIRED_COLUMNS, TARGET,
)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_features(df: pd.DataFrame) -> ValidationResult:
    """Validate a frame for PREDICTION (target not required)."""
    errors: list[str] = []
    warnings: list[str] = []
    if df is None or len(df) == 0:
        return ValidationResult(False, ["The uploaded file contains no rows."])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            bad = int((converted.isna() & df[col].notna()).sum())
            if bad:
                errors.append(f"Column '{col}' has {bad} non-numeric value(s).")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isna().all():
            errors.append(f"Column '{col}' is entirely empty.")

    if "Age" in df.columns:
        age = pd.to_numeric(df["Age"], errors="coerce")
        if ((age < 0) | (age > 120)).any():
            warnings.append("Some 'Age' values are outside 0-120.")
    if "MonthlyIncome" in df.columns:
        inc = pd.to_numeric(df["MonthlyIncome"], errors="coerce")
        if (inc < 0).any():
            errors.append("'MonthlyIncome' contains negative values.")

    return ValidationResult(not errors, errors, warnings)


def validate_training(df: pd.DataFrame) -> ValidationResult:
    """Validate a frame for TRAINING (target required and binary)."""
    res = validate_features(df)
    if TARGET not in df.columns:
        res.errors.append(f"Target column '{TARGET}' is missing.")
        res.ok = False
        return res
    labels = set(df[TARGET].dropna().unique().tolist())
    if not labels.issubset({0, 1}):
        res.errors.append(
            f"Target '{TARGET}' must contain only 0/1; found {sorted(map(str, labels))}."
        )
    if df[TARGET].isna().any():
        res.errors.append(f"Target '{TARGET}' has missing values.")
    if len(labels) < 2 and not res.errors:
        res.errors.append("Target has only one class; cannot train a classifier.")
    if len(df) < 50:
        res.errors.append("At least 50 rows are required to train.")
    res.ok = not res.errors
    return res


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Light, deterministic cleaning that does not learn from data.

    Fixes known typos in the Tourism dataset (e.g. 'Fe Male'), strips
    whitespace, and coerces numeric columns.
    """
    out = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype("object").where(out[col].notna(), None)
            out[col] = out[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    if "Gender" in out.columns:
        out["Gender"] = out["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in out.columns:
        out["MaritalStatus"] = out["MaritalStatus"].replace({"Unmarried": "Single"})
    for col in NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if TARGET in out.columns:
        out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    return out


def quality_report(df: pd.DataFrame) -> dict:
    """Summary statistics for the Data Explorer page."""
    rep: dict = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
    }
    if TARGET in df.columns:
        vc = df[TARGET].value_counts(dropna=False)
        rep["class_balance"] = {str(k): int(v) for k, v in vc.items()}
        rep["positive_rate"] = (
            float((df[TARGET] == 1).mean()) if len(df) else 0.0
        )
    return rep
