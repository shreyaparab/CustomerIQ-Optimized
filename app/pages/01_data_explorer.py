import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.state import require_data, setup  # noqa: E402
from src.data.validation import quality_report, validate_training  # noqa: E402
from configs.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET  # noqa: E402

setup("Data Explorer", "🔎")
df = require_data()
rep = quality_report(df)

a, b, c, d = st.columns(4)
a.metric("Rows", f"{rep['rows']:,}")
b.metric("Columns", rep["columns"])
c.metric("Duplicate rows", rep["duplicates"])
d.metric("Purchase rate", f"{rep.get('positive_rate', 0):.1%}")

res = validate_training(df)
for w in res.warnings:
    st.warning(w)
for e in res.errors:
    st.error(e)
if res.ok:
    st.success("Schema validation passed.")

st.subheader("Missing values")
if rep["missing"]:
    st.bar_chart(rep["missing"])
else:
    st.caption("No missing values.")

st.subheader("Target balance")
st.write(rep.get("class_balance", {}))
st.caption("ProdTaken = 1 means the customer purchased the package. "
           "CustomerID is excluded from modelling.")

st.subheader("Distributions")
col = st.selectbox("Column", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
if col in NUMERIC_FEATURES:
    fig = px.histogram(df, x=col, color=df[TARGET].astype(str), barmode="overlay", nbins=40)
else:
    fig = px.histogram(df, x=col, color=df[TARGET].astype(str), barmode="group")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Preview")
st.dataframe(df.head(100), use_container_width=True)
