"""Synthetic sample data in the Holiday Package (Tourism) schema.

FOR DEMOS AND TESTS ONLY. It is NOT the real dataset; model metrics computed on
it say nothing about real customers. Replace with the real Tourism.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_sample(n: int = 4888, seed: int = 42, with_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 62, n).astype(float)
    income = np.clip(rng.normal(23000, 5500, n) + (age - 35) * 150, 8000, 98000).round()
    designation = np.where(income < 18000, "Executive",
                  np.where(income < 24000, "Manager",
                  np.where(income < 30000, "Senior Manager",
                  np.where(income < 36000, "AVP", "VP"))))
    product = np.select(
        [designation == "Executive", designation == "Manager",
         designation == "Senior Manager", designation == "AVP"],
        ["Basic", "Deluxe", "Standard", "Super Deluxe"], "King")
    passport = rng.binomial(1, 0.29, n)
    trips = np.clip(rng.poisson(3.2, n) + 1, 1, 22).astype(float)
    df = pd.DataFrame({
        "CustomerID": np.arange(200000, 200000 + n),
        "Age": age,
        "TypeofContact": rng.choice(["Self Enquiry", "Company Invited"], n, p=[0.7, 0.3]),
        "CityTier": rng.choice([1, 2, 3], n, p=[0.65, 0.05, 0.30]).astype(float),
        "DurationOfPitch": np.clip(rng.normal(15.5, 8, n), 5, 127).round(),
        "Occupation": rng.choice(["Salaried", "Small Business", "Large Business", "Free Lancer"],
                                 n, p=[0.48, 0.42, 0.09, 0.01]),
        "Gender": rng.choice(["Male", "Female"], n, p=[0.6, 0.4]),
        "NumberOfPersonVisiting": rng.choice([1, 2, 3, 4, 5], n, p=[0.01, 0.29, 0.48, 0.21, 0.01]).astype(float),
        "NumberOfFollowups": rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.08, 0.1, 0.43, 0.28, 0.09, 0.02]).astype(float),
        "ProductPitched": product,
        "PreferredPropertyStar": rng.choice([3, 4, 5], n, p=[0.62, 0.2, 0.18]).astype(float),
        "MaritalStatus": rng.choice(["Married", "Divorced", "Single"], n, p=[0.48, 0.19, 0.33]),
        "NumberOfTrips": trips,
        "Passport": passport,
        "PitchSatisfactionScore": rng.choice([1, 2, 3, 4, 5], n, p=[0.19, 0.1, 0.31, 0.2, 0.2]),
        "OwnCar": rng.binomial(1, 0.62, n),
        "NumberOfChildrenVisiting": rng.choice([0, 1, 2, 3], n, p=[0.1, 0.43, 0.38, 0.09]).astype(float),
        "Designation": designation,
        "MonthlyIncome": income,
    })
    # Inject realistic missingness
    for col, frac in [("Age", 0.05), ("TypeofContact", 0.005), ("DurationOfPitch", 0.05),
                      ("NumberOfFollowups", 0.01), ("PreferredPropertyStar", 0.005),
                      ("NumberOfTrips", 0.03), ("NumberOfChildrenVisiting", 0.01),
                      ("MonthlyIncome", 0.05)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    if with_target:
        z = (-2.1 + 1.3 * passport + 0.5 * (df["ProductPitched"] == "Basic")
             - 0.03 * (df["Age"].fillna(35) - 35) + 0.25 * (df["PitchSatisfactionScore"] - 3)
             + 0.2 * (df["NumberOfFollowups"].fillna(3) - 3)
             + 0.4 * (df["MaritalStatus"] == "Single") - 0.3 * (df["CityTier"] == 2)
             + rng.normal(0, 0.5, n))
        p = 1 / (1 + np.exp(-z))
        df["ProdTaken"] = rng.binomial(1, p)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic sample CSV")
    ap.add_argument("--rows", type=int, default=4888)
    ap.add_argument("--out", default="data/sample_tourism.csv")
    ap.add_argument("--no-target", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_sample(a.rows, with_target=not a.no_target).to_csv(out, index=False)
    print(f"Wrote {out} ({a.rows} rows)")


if __name__ == "__main__":
    main()
