import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sample_data import make_sample  # noqa: E402
from src.data.validation import clean_dataframe  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return clean_dataframe(make_sample(900, seed=1))


@pytest.fixture(scope="session")
def trained(df):
    from src.models.train import train_and_compare
    return train_and_compare(df, cv_folds=3)
