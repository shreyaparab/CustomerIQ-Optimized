import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.state import require_data, setup  # noqa: E402
from src.data.validation import validate_training  # noqa: E402
from src.models.predict import save_model  # noqa: E402
from src.models.train import HAS_XGB, train_and_compare  # noqa: E402
from src.storage.repository import log_event  # noqa: E402

setup("Model Lab", "🧪")
df = require_data()

res = validate_training(df)
if not res.ok:
    for e in res.errors:
        st.error(e)
    st.stop()

st.write("Models are compared with stratified cross-validation on the training split. "
         "The untouched test set is evaluated once, for the selected model only. "
         "The decision threshold is chosen on out-of-fold training predictions.")
if not HAS_XGB:
    st.warning("XGBoost is not installed; it will be skipped. `pip install xgboost` to include it.")

c1, c2 = st.columns(2)
folds = c1.slider("CV folds", 3, 10, 5)
metric = c2.selectbox("Selection metric", ["pr_auc", "roc_auc", "f1"], index=0)

if st.button("Train and compare models", type="primary"):
    with st.spinner("Training models..."):
        try:
            result = train_and_compare(df, cv_folds=folds, selection_metric=metric)
            path = save_model(result)
            log_event("model_trained", f"{result.best_name} {result.metadata['model_version']}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.stop()
    st.session_state["last_result"] = result
    st.success(f"Best model: {result.best_name} (saved to {path.name})")

result = st.session_state.get("last_result")
if result is not None:
    st.subheader("Cross-validation comparison")
    st.dataframe(result.comparison.round(4), use_container_width=True)

    m = result.test_metrics
    st.subheader(f"Held-out test results — {result.best_name}")
    a, b, c, d, e = st.columns(5)
    a.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    b.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c.metric("Precision", f"{m['precision']:.3f}")
    d.metric("Recall", f"{m['recall']:.3f}")
    e.metric("F1", f"{m['f1']:.3f}")
    st.caption(f"Threshold {m['threshold']:.3f} (chosen on training OOF data to maximise F1). "
               f"Brier score {m['brier']:.3f}. Support: {m['support_pos']} positive, "
               f"{m['support_neg']} negative.")

    cm = m["confusion_matrix"]
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                               text=cm, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(title="Confusion matrix", height=350)
    st.plotly_chart(fig, use_container_width=True)

    cal = m["calibration"]
    if cal["mean_predicted"]:
        f2 = go.Figure()
        f2.add_scatter(x=cal["mean_predicted"], y=cal["fraction_positive"],
                       mode="lines+markers", name="Model")
        f2.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash"))
        f2.update_layout(title="Calibration", xaxis_title="Mean predicted",
                         yaxis_title="Observed rate", height=350)
        st.plotly_chart(f2, use_container_width=True)
    st.caption("Accuracy alone is misleading when buyers are a minority; compare against the Dummy baseline.")
