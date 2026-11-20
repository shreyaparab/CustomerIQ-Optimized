"""Shared Streamlit session helpers (keep UI glue out of the pages)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.data.sample_data import make_sample  # noqa: E402
from src.data.validation import clean_dataframe, validate_features, validate_training  # noqa: E402


def setup(title: str, icon: str = "📊") -> None:
    st.set_page_config(page_title=f"CustomerIQ · {title}", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")


def get_training_df() -> pd.DataFrame | None:
    return st.session_state.get("train_df")


def dataset_loader(key: str = "loader") -> pd.DataFrame | None:
    """Sidebar uploader shared by pages that need the training data."""
    with st.sidebar:
        st.subheader("Dataset")
        up = st.file_uploader("Upload training CSV (with ProdTaken)", type=["csv"], key=key)
        if st.button("Use synthetic sample data", key=key + "_sample"):
            st.session_state["train_df"] = clean_dataframe(make_sample(4888))
            st.session_state["train_source"] = "synthetic sample (demo only)"
        if up is not None:
            try:
                df = clean_dataframe(pd.read_csv(up))
            except Exception as exc:  # unreadable file
                st.error(f"Could not read the file: {exc}")
            else:
                res = validate_training(df)
                for w in res.warnings:
                    st.warning(w)
                if res.ok:
                    st.session_state["train_df"] = df
                    st.session_state["train_source"] = up.name
                else:
                    for e in res.errors:
                        st.error(e)
        if st.session_state.get("train_df") is not None:
            st.success(f"Loaded: {st.session_state.get('train_source')}")
    return st.session_state.get("train_df")


def require_data() -> pd.DataFrame:
    df = dataset_loader()
    if df is None:
        st.info("Upload a training CSV or click **Use synthetic sample data** in the sidebar.")
        st.stop()
    return df
