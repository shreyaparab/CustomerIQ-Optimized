import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.state import setup  # noqa: E402
from src.optimization.budget_optimizer import optimize_campaign  # noqa: E402
from src.storage.repository import log_event  # noqa: E402

setup("Campaign Planner", "💰")
preds = st.session_state.get("preds")
if preds is None:
    st.info("Score customers on the **Customer Predictions** page first.")
    st.stop()

st.write("**Formula:** expected net value = probability × assumed margin − contact cost. "
         "Results are **scenario estimates** from purchase propensity, not measured profit "
         "or causal uplift.")

def scenario_inputs(prefix: str, defaults: dict) -> dict:
    a, b, c, d = st.columns(4)
    return {
        "budget": a.number_input("Budget", 0.0, 1e9, defaults["budget"], 100.0, key=prefix + "b"),
        "margin": b.number_input("Margin per purchase", 0.0, 1e9, defaults["margin"], 10.0, key=prefix + "m"),
        "cost": c.number_input("Cost per contact", 0.0, 1e6, defaults["cost"], 1.0, key=prefix + "c"),
        "max_contacts": d.number_input("Max contacts (0 = no limit)", 0, 1_000_000, 0, key=prefix + "x"),
    }

def run(s: dict):
    return optimize_campaign(preds, s["budget"], s["margin"], s["cost"],
                             s["max_contacts"] or None)

st.subheader("Scenario A")
sa = scenario_inputs("a", {"budget": 2000.0, "margin": 300.0, "cost": 20.0})
compare = st.checkbox("Compare with a second scenario")
if compare:
    st.subheader("Scenario B")
    sb = scenario_inputs("b", {"budget": 4000.0, "margin": 300.0, "cost": 20.0})

if st.button("Run optimizer", type="primary"):
    try:
        results = {"A": run(sa)}
        if compare:
            results["B"] = run(sb)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    log_event("campaign_optimized", f"budget={sa['budget']}")
    summary = pd.DataFrame([{
        "scenario": k, "budget": r.budget, "contacts": r.n_selected,
        "total_cost": r.total_cost, "expected_purchases": round(r.expected_purchases, 2),
        "expected_net_value": round(r.expected_net_value, 2)} for k, r in results.items()])
    st.dataframe(summary, use_container_width=True)
    for k, r in results.items():
        st.subheader(f"Selected customers — scenario {k}")
        if r.n_selected == 0:
            st.warning("No customers have positive expected net value under these assumptions.")
        else:
            st.dataframe(r.selected, use_container_width=True)
            st.download_button(f"Download target list {k}", r.selected.to_csv(index=False).encode(),
                               f"target_list_{k}.csv", "text/csv", key=f"dl{k}")
    st.caption("Assumptions: " + str(results["A"].assumptions["note"]))
