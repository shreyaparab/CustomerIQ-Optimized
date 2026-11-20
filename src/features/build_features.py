"""Convenience loader that cleans and validates a CSV for the app and CLI."""
from __future__ import annotations

import pandas as pd

from src.data.validation import clean_dataframe, validate_features, validate_training


def load_csv(file_or_path, training: bool = False) -> tuple[pd.DataFrame, object]:
    df = pd.read_csv(file_or_path)
    df = clean_dataframe(df)
    res = validate_training(df) if training else validate_features(df)
    return df, res
