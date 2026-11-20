"""Command-line training: python train_cli.py --data data/Tourism.csv"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.sample_data import make_sample  # noqa: E402
from src.data.validation import clean_dataframe, validate_training  # noqa: E402
from src.models.predict import save_model  # noqa: E402
from src.models.train import train_and_compare  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Training CSV with ProdTaken")
    ap.add_argument("--sample", action="store_true", help="Use synthetic demo data")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    if a.sample:
        df = clean_dataframe(make_sample(4888))
        print("NOTE: using synthetic data; metrics are not real-world results.")
    elif a.data:
        df = clean_dataframe(pd.read_csv(a.data))
    else:
        ap.error("provide --data FILE or --sample")
    res = validate_training(df)
    if not res.ok:
        print("Validation failed:\n- " + "\n- ".join(res.errors))
        return 1
    result = train_and_compare(df, cv_folds=a.folds)
    path = save_model(result)
    print(result.comparison.round(4).to_string(index=False))
    m = result.test_metrics
    print(f"\nBest: {result.best_name} | test ACC {m["accuracy"]:.4f} PR-AUC {m['pr_auc']:.3f} "
          f"ROC-AUC {m['roc_auc']:.3f} F1 {m['f1']:.3f}\nSaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
