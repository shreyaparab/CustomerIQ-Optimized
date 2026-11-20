import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.state import require_data, setup  # noqa: E402
from configs.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET  # noqa: E402
from src.explainability.shap_reports import global_importance, local_explanation  # noqa: E402
from src.models.predict import load_model  # noqa: E402

setup("Explainability", "💡")
df = require_data()
try:
    pipeline, meta = load_model()
except FileNotFoundError as exc:
    st.info(str(exc))
    st.stop()

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET].astype(int)

st.subheader("Global feature influence")
with st.spinner("Computing..."):
    try:
        imp, method = global_importance(pipeline, X, y)
    except Exception as exc:
        st.error(f"Could not compute importance: {exc}")
        st.stop()
st.caption(f"Method: {method}")
st.plotly_chart(px.bar(imp.head(15).iloc[::-1], x="importance", y="feature", orientation="h"),
                use_container_width=True)

st.subheader("Individual customer")
idx = st.number_input("Row index", 0, len(df) - 1, 0)
row = X.iloc[[int(idx)]]
prob = float(pipeline.predict_proba(row)[0, 1])
st.metric("Purchase probability", f"{prob:.1%}")
try:
    local, lmethod = local_explanation(pipeline, row, X)
    st.caption(f"Method: {lmethod}")
    st.plotly_chart(px.bar(local.iloc[::-1], x="contribution", y="feature", orientation="h"),
                    use_container_width=True)
    st.dataframe(local, use_container_width=True)
except Exception as exc:
    st.error(f"Could not explain this prediction: {exc}")

st.info("These explanations show what the model associates with purchase. "
        "They do not prove causation or fairness.")
