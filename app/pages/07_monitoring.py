import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.components.state import setup  # noqa: E402
from src.monitoring.drift import drift_report  # noqa: E402
from src.storage.repository import list_events, list_prediction_runs  # noqa: E402

setup("Monitoring & Audit", "📈")

st.subheader("Data drift (training vs latest scored batch)")
ref = st.session_state.get("train_df")
cur = st.session_state.get("scored_df")
if ref is None or cur is None:
    st.info("Load training data and score a batch to compute drift.")
else:
    rep = drift_report(ref, cur)
    st.dataframe(rep, use_container_width=True)
    st.caption("Numeric: PSI (>0.1 moderate, >0.25 high). Categorical: total variation distance.")

st.subheader("Prediction run history")
runs = list_prediction_runs()
st.dataframe(runs, use_container_width=True) if runs else st.caption("No runs yet.")

st.subheader("Audit events")
ev = list_events()
st.dataframe(ev, use_container_width=True) if ev else st.caption("No events yet.")
st.caption("Only run-level metadata is stored; raw customer values are not logged. "
           "Performance metrics require outcome labels to arrive later.")
