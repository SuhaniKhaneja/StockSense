"""Conformal calibration of the model's nominal 80% prediction interval.

The gradient-boosting 10th/90th percentile models give an interval that covers about
92% of outcomes (too wide). Demand cannot go below zero and the lower bound sits at zero
on most days, so a symmetric correction would push it above zero and miss every
zero-demand day. Instead:

  - the lower bound is left unchanged;
  - the upper bound moves by q * width, where width = max(hi - lo, 1) and q comes from split
    conformal prediction on the scores (y - hi) / width, so the shift scales with how wide
    (roughly: how high-volume) each interval is. A single fixed shift in units does almost
    nothing when a few high-volume products dominate the average width;
    the miss budget is (alpha - share of calibration outcomes already below the lower bound);
  - the point forecast always stays inside the interval.

Evaluation is out-of-sample: fold f is calibrated only on folds older than f.
The deployed offset is calibrated on all folds.

Usage:  python -m stocksense.calibrate .\\data\\sales.csv
Writes: artifacts/interval_calibration.csv and artifacts/interval_calibration.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.2                 # 80% interval
MODEL = "Gradient boosting"
MIN_BUDGET = 0.02           # never ask the upper tail to absorb less than this


def coverage(y, lo, hi):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def _width(lo, hi):
    return np.maximum(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float), 1.0)


def upper_offset(y, lo, hi, alpha=ALPHA):
    """Relative offset q (in units of interval width) so that hi + q * width has about
    (1 - alpha) total coverage with `lo` unchanged."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    n = len(y)
    miss_lo = float(np.mean(y < lo))
    budget = max(alpha - miss_lo, MIN_BUDGET)
    level = min(1.0, np.ceil((n + 1) * (1 - budget)) / n)
    return float(np.quantile((y - hi) / _width(lo, hi), level, method="higher"))


def apply_offset(p, lo, hi, q):
    """Keep lo, shift hi by q * width, keep the point forecast inside, never below zero."""
    p, lo, hi = (np.asarray(a, dtype=float) for a in (p, lo, hi))
    return np.maximum(np.minimum(lo, p), 0.0), np.maximum(hi + q * _width(lo, hi), p)


def evaluate(preds, alpha=ALPHA):
    """Out-of-sample before/after coverage. Fold 0 is the newest; the oldest fold is only
    ever used for calibration."""
    g = preds[preds["model"] == MODEL].dropna(subset=["lo", "hi"])
    folds = sorted(g["fold"].unique())
    rows, pooled = [], []
    for f in folds[:-1]:
        cal, test = g[g["fold"] > f], g[g["fold"] == f]
        q = upper_offset(cal["y"], cal["lo"], cal["hi"], alpha)
        lo2, hi2 = apply_offset(test["p"], test["lo"], test["hi"], q)
        rows.append({"fold": f, "calibrated_on_folds": f"{f + 1}-{folds[-1]}", "offset_rel_width": q,
                     "coverage_before": coverage(test["y"], test["lo"], test["hi"]),
                     "coverage_after": coverage(test["y"], lo2, hi2),
                     "width_before": float((test["hi"] - test["lo"]).mean()),
                     "width_after": float((hi2 - lo2).mean())})
        pooled.append(test.assign(lo2=lo2, hi2=hi2))
    allp = pd.concat(pooled)
    rows.append({"fold": "ALL", "calibrated_on_folds": "older folds only",
                 "offset_rel_width": float(np.mean([r["offset_rel_width"] for r in rows])),
                 "coverage_before": coverage(allp["y"], allp["lo"], allp["hi"]),
                 "coverage_after": coverage(allp["y"], allp["lo2"], allp["hi2"]),
                 "width_before": float((allp["hi"] - allp["lo"]).mean()),
                 "width_after": float((allp["hi2"] - allp["lo2"]).mean())})
    return pd.DataFrame(rows)


def deployed_offset(preds, alpha=ALPHA):
    g = preds[preds["model"] == MODEL].dropna(subset=["lo", "hi"])
    return upper_offset(g["y"], g["lo"], g["hi"], alpha), len(g)


def main(path):
    from .backtest import backtest

    df = pd.read_csv(path, parse_dates=["date"])
    _, _, preds = backtest(df)
    table = evaluate(preds)
    q, n = deployed_offset(preds)

    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print(f"\nDeployed upper-bound offset (calibrated on all {n} forecasts): {q:+.3f} x interval width")
    Path("artifacts").mkdir(exist_ok=True)
    table.round(4).to_csv("artifacts/interval_calibration.csv", index=False)
    json.dump({"alpha": ALPHA, "lower_bound": "unchanged", "upper_offset_rel_width": round(q, 5),
               "n_calibration": n}, open("artifacts/interval_calibration.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv")
