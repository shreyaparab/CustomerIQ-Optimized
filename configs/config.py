"""Central configuration. No secrets live here; use environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.getenv("CUSTOMERIQ_ARTIFACT_DIR", ROOT / "artifacts"))
DB_PATH = Path(os.getenv("CUSTOMERIQ_DB_PATH", ROOT / "data" / "customeriq.db"))
SEED = int(os.getenv("CUSTOMERIQ_RANDOM_SEED", "42"))

TARGET = "ProdTaken"
ID_COLUMNS = ["CustomerID"]

# Columns of the Holiday Package (Tourism) dataset.
NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features used for K-Means (no target, no ID).
CLUSTER_FEATURES = [
    "Age", "MonthlyIncome", "NumberOfTrips", "NumberOfPersonVisiting",
    "NumberOfChildrenVisiting", "CityTier", "Passport", "OwnCar",
]
