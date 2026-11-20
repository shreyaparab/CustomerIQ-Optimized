# CustomerIQ - Optimized Version: what changed

Dataset: Tourism.csv (4,128 rows, 19.3% buyers). Same 80/20 stratified split, seed 42.

| Metric (held-out test) | Previous (best model: XGBoost/RF-300 class) | Optimized (Ensemble) |
|---|---|---|
| Accuracy | ~0.907 (RF-300, 0.5 threshold) | **0.955** |
| ROC-AUC | 0.977 | **0.986** |
| PR-AUC | 0.916 | **0.949** |
| F1 (buyers) | 0.696 | **0.884** |

## Changes
1. New model candidates: Extra Trees, Hist Gradient Boosting, and a soft-voting Ensemble (ET + HGB + RF).
2. Random Forest relaxed (min_samples_leaf 1, no class-weight): the old settings hurt recall on this data.
3. Accuracy added to CV comparison and test metrics.
4. Model chosen by 5-fold CV PR-AUC on training data; threshold tuned on out-of-fold predictions; test set used once (no leakage).
5. Real data/Tourism.csv bundled; `python train_cli.py --data data/Tourism.csv`.
Tried and rejected: engineered ratio features (lowered test scores).
