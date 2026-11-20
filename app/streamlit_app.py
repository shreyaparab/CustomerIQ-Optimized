"""CustomerIQ — Overview page (Streamlit entry point)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.components.state import dataset_loader, setup  # noqa: E402
from src.data.validation import quality_report  # noqa: E402
from src.models.predict import load_model  # noqa: E402
from src.storage.repository import list_prediction_runs  # noqa: E402

setup("Overview", "🏠")
st.caption("Customer intelligence and marketing decision support. "
           "Estimates are scenario-based, not proven campaign impact.")

df = dataset_loader()
c1, c2, c3 = st.columns(3)
if df is not None:
    rep = quality_report(df)
    c1.metric("Rows", f"{rep['rows']:,}")
    c2.metric("Columns", rep["columns"])
    c3.metric("Purchase rate", f"{rep.get('positive_rate', 0):.1%}")
    st.caption(f"Source: {st.session_state.get('train_source')}")
else:
    st.info("No dataset loaded yet. Use the sidebar to upload a CSV or load the synthetic sample.")

st.subheader("Latest model")
try:
    _, meta = load_model()
    st.write({k: meta[k] for k in ("model_version", "model_name", "threshold", "trained_at_utc")})
except FileNotFoundError:
    st.info("No model trained yet. Open **Model Lab** to train one.")

st.subheader("Recent prediction runs")
runs = list_prediction_runs(10)
if runs:
    st.dataframe(runs, use_container_width=True)
else:
    st.caption("No prediction runs recorded yet.")

st.markdown("""
**Workflow:** Data Explorer → Model Lab → Customer Predictions → Segments →
Explainability → Campaign Planner → Monitoring.
""")
