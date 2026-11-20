import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.state import setup  # noqa: E402
from configs.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402
from src.data.sample_data import make_sample  # noqa: E402
from src.data.validation import clean_dataframe, validate_features  # noqa: E402
from src.models.predict import load_model, predict_batch  # noqa: E402
from src.storage.repository import log_prediction_run  # noqa: E402

setup("Customer Predictions", "🎯")

try:
    pipeline, meta = load_model()
except FileNotFoundError as exc:
    st.info(str(exc))
    st.stop()

st.caption(f"Model {meta['model_name']} · version {meta['model_version']}")
thr = st.slider("Decision threshold", 0.05, 0.95, float(meta["threshold"]), 0.01,
                help="Chosen on training data to maximise F1 by default. "
                     "0.5 is not universally optimal; adjust to your business costs.")

tab1, tab2 = st.tabs(["Batch (CSV)", "Single customer"])

with tab1:
    up = st.file_uploader("Customer CSV for scoring", type=["csv"])
    if st.button("Use synthetic sample (no labels)"):
        st.session_state["score_df"] = clean_dataframe(make_sample(500, seed=7, with_target=False))
    if up is not None:
        try:
            st.session_state["score_df"] = clean_dataframe(pd.read_csv(up))
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
    df = st.session_state.get("score_df")
    if df is not None:
        res = validate_features(df)
        for w in res.warnings:
            st.warning(w)
        if not res.ok:
            for e in res.errors:
                st.error(e)
        else:
            preds = predict_batch(pipeline, meta, df, threshold=thr)
            st.session_state["preds"] = preds
            st.session_state["scored_df"] = df
            run_key = (meta["model_version"], len(df), round(thr, 3),
                       float(preds["purchase_probability"].sum()))
            if st.session_state.get("last_run_key") != run_key:  # log once per distinct run
                log_prediction_run(meta["model_version"], len(preds),
                                   float(preds["purchase_probability"].mean()),
                                   float(preds["predicted_label"].mean()), list(df.columns))
                st.session_state["last_run_key"] = run_key
            a, b, c = st.columns(3)
            a.metric("Customers scored", f"{len(preds):,}")
            b.metric("Flagged as likely buyers", int(preds["predicted_label"].sum()))
            c.metric("Mean probability", f"{preds['purchase_probability'].mean():.3f}")
            st.plotly_chart(px.histogram(preds, x="purchase_probability", nbins=30),
                            use_container_width=True)
            st.dataframe(preds.head(200), use_container_width=True)
            st.download_button("Download predictions CSV",
                               preds.to_csv(index=False).encode(), "predictions.csv", "text/csv")
            st.caption("Probabilities are model scores; check calibration in Model Lab. "
                       "Exports contain customer IDs only, not other personal fields.")

with tab2:
    st.write("Enter one customer's details.")
    cols = st.columns(3)
    vals = {}
    defaults_num = {"Age": 35, "CityTier": 1, "DurationOfPitch": 15, "NumberOfPersonVisiting": 3,
                    "NumberOfFollowups": 3, "PreferredPropertyStar": 3, "NumberOfTrips": 3,
                    "Passport": 0, "PitchSatisfactionScore": 3, "OwnCar": 1,
                    "NumberOfChildrenVisiting": 1, "MonthlyIncome": 22000}
    for i, f in enumerate(NUMERIC_FEATURES):
        vals[f] = cols[i % 3].number_input(f, value=float(defaults_num[f]), key=f"n_{f}")
    options = {
        "TypeofContact": ["Self Enquiry", "Company Invited"],
        "Occupation": ["Salaried", "Small Business", "Large Business", "Free Lancer"],
        "Gender": ["Male", "Female"],
        "ProductPitched": ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
        "MaritalStatus": ["Married", "Single", "Divorced"],
        "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    }
    for i, f in enumerate(CATEGORICAL_FEATURES):
        vals[f] = cols[i % 3].selectbox(f, options[f], key=f"c_{f}")
    if st.button("Score customer"):
        p = predict_batch(pipeline, meta, pd.DataFrame([vals]), threshold=thr)
        prob = float(p["purchase_probability"].iloc[0])
        st.metric("Purchase probability", f"{prob:.1%}")
        st.write("Predicted: " + ("likely buyer" if prob >= thr else "unlikely buyer")
                 + f" (threshold {thr:.2f})")
