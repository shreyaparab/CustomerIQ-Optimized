# CustomerIQ — Customer Intelligence & Revenue Optimization

Streamlit app for purchase prediction, customer segmentation, explainability and
budget-constrained campaign selection on the Holiday Package (Tourism) schema.

## Quick start (Windows cmd)

```
cd customeriq
setup_windows.bat
```

Or manually:

```
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_cli.py --sample
python -m pytest -q
streamlit run app\streamlit_app.py
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && streamlit run app/streamlit_app.py`

## Using the real dataset
Put the Holiday Package Purchase `Tourism.csv` (target column `ProdTaken`, ID column
`CustomerID`) in `data/` and upload it in the sidebar, or run
`python train_cli.py --data data\Tourism.csv`. Check the dataset license first.
The bundled generator (`python -m src.data.sample_data`) makes **synthetic** data for
demos/tests only — metrics on it say nothing about real customers.

## Demo flow
1. Sidebar → *Use synthetic sample data* (or upload CSV) → Data Explorer.
2. Model Lab → *Train and compare models* (Dummy, Logistic, Tree, Random Forest, Extra Trees, Hist Gradient Boosting, a soft-voting Ensemble, and XGBoost if installed).
3. Customer Predictions → score a CSV or one customer; download results.
4. Segments (K-Means + PCA), Explainability (SHAP or permutation fallback).
5. Campaign Planner → enter budget, margin, cost; compare two scenarios; export target list.
6. Monitoring → drift (PSI), run history, audit events.

## Design notes
- Split first; preprocessing lives inside a Pipeline so it is fitted on training folds only.
- Threshold is chosen on out-of-fold training predictions; test set is used once.
- Optimizer: exact 0/1 knapsack in integer cents (greedy fallback for huge inputs / contact caps);
  budget is never exceeded. Outputs are **scenario estimates** (p × margin − cost), not
  measured profit or causal uplift.
- SHAP and XGBoost are optional; the app falls back gracefully if missing.
- Only run-level metadata is stored in SQLite; no raw customer values.

## Limitations
Purchase propensity ≠ incremental campaign impact. Cluster labels are descriptive and need human review.
Explanations show associations, not causation or fairness. Keep a human in the loop before contacting customers.

## Tests
`python -m pytest -q` (44 tests: validation, preprocessing leakage safeguards, prediction, optimizer constraints, segmentation, drift, storage).
