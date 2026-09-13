"""Aggregate evaluation metrics across seeds into mean +/- standard deviation.

A single training run is not a result. RAF-DB's rare classes are tiny in the
test split — Fear has 74 images, so one flipped prediction moves its recall by
1.35 points — and run-to-run variance easily exceeds the differences people
report between methods. Reporting one number from one seed overstates precision.

Reads the JSON files evaluate.py writes, one per seed, and prints a table ready
to paste into a paper alongside the sample size.

Usage:
    # train and evaluate each seed with a tag
    python train.py    --seed 42 --tag seed42
    python evaluate.py --tag seed42
    ...repeat for 43, 44...

    python aggregate_seeds.py
    python aggregate_seeds.py --glob 'eval_metrics_seed*_test_tta.json'
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rafdb.core import config as C

OVERALL_KEYS = [
    ("accuracy", "accuracy"),
    ("balanced_accuracy", "balanced accuracy"),
    ("macro_f1", "macro-F1"),
    ("weighted_f1", "weighted-F1"),
    ("macro_precision", "macro-precision"),
    ("macro_recall", "macro-recall"),
]


def load_runs(out_dir: Path, pattern: str) -> list[dict]:
    """Load every metrics JSON matching `pattern`, newest-agnostic order."""
    paths = sorted(out_dir.glob(pattern))
    runs = []
    for path in paths:
        with open(path) as f:
            payload = json.load(f)
        payload["_file"] = path.name
        runs.append(payload)
    return runs


def mean_std(values) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1); std is 0.0 for a single run."""
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate metrics across seeds")
    ap.add_argument("--glob", default="eval_metrics_seed*_test.json",
                    help="filename pattern inside outputs/ "
                         "(default: eval_metrics_seed*_test.json)")
    args = ap.parse_args()

    runs = load_runs(C.OUT_DIR, args.glob)
    if not runs:
        raise SystemExit(
            f"No metrics files matched {args.glob!r} in {C.OUT_DIR}.\n"
            "Train and evaluate with matching --tag values first, e.g.\n"
            "  python train.py --seed 42 --tag seed42 && python evaluate.py --tag seed42"
        )

    seeds = [r.get("seed") for r in runs]
    splits = {r.get("split") for r in runs}
    ttas = {r.get("tta") for r in runs}
    print("=" * 66)
    print(f"AGGREGATE OVER {len(runs)} RUN(S)")
    print("=" * 66)
    print(f"  files : {', '.join(r['_file'] for r in runs)}")
    print(f"  seeds : {seeds}")
    print(f"  split : {', '.join(str(s) for s in splits)}"
          f"   TTA: {', '.join(str(t) for t in ttas)}")
    if len(splits) > 1 or len(ttas) > 1:
        print("  WARNING: mixing splits or TTA settings — the mean is meaningless.")
    if len(runs) < 3:
        print(f"  NOTE: {len(runs)} run(s) is too few for a credible spread; "
              "3+ seeds recommended.")

    print("\n" + "-" * 66)
    print(f"{'metric':<22} {'mean':>9} {'std':>9}   {'per-seed'}")
    print("-" * 66)
    for key, label in OVERALL_KEYS:
        vals = [r["overall"][key] for r in runs]
        mu, sd = mean_std(vals)
        per = " ".join(f"{v:.4f}" for v in vals)
        print(f"{label:<22} {mu:>9.4f} {sd:>9.4f}   {per}")

    print("\n" + "-" * 66)
    print(f"{'per-class recall':<22} {'mean':>9} {'std':>9}   {'support':>8}")
    print("-" * 66)
    for name in C.EMOTION_NAMES:
        vals = [r["per_class"][name]["recall"] for r in runs]
        mu, sd = mean_std(vals)
        support = runs[0]["per_class"][name]["support"]
        print(f"{name:<22} {mu:>9.4f} {sd:>9.4f}   {support:>8}")

    # Paper-ready one-liner.
    acc_mu, acc_sd = mean_std([r["overall"]["accuracy"] for r in runs])
    f1_mu, f1_sd = mean_std([r["overall"]["macro_f1"] for r in runs])
    print("\n" + "=" * 66)
    print("FOR THE WRITEUP")
    print("=" * 66)
    print(f"  accuracy {acc_mu*100:.2f} +/- {acc_sd*100:.2f} %, "
          f"macro-F1 {f1_mu:.4f} +/- {f1_sd:.4f}  (n={len(runs)} seeds)")
    print("  Model selection used a validation split held out of the training")
    print("  partition; the test partition was evaluated once per seed.")


if __name__ == "__main__":
    main()
