"""Daily vs weekly forecast error, reusing the existing backtest predictions.

Usage:  python -m stocksense.weekly .\\data\\sales.csv
Needs backtest() to return `fold` and `date` columns in its predictions.
Writes: artifacts/weekly_wape.csv
"""
import sys
from pathlib import Path

import pandas as pd

from .segments import classify_skus, wape_by_segment


def to_weekly(preds):
    """Sum y and p over 7-day blocks inside each fold (aligned to the fold start,
    so no partial weeks). Returns columns: model, sku, y, p."""
    d = preds.copy()
    d["block"] = (d["date"] - d.groupby("fold")["date"].transform("min")).dt.days // 7
    return (d.groupby(["model", "fold", "block", "sku"], as_index=False)[["y", "p"]].sum())


def main(path):
    from .backtest import backtest, HORIZON

    df = pd.read_csv(path, parse_dates=["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=4 * HORIZON)
    segs = classify_skus(df, until=cutoff)
    _, _, preds = backtest(df)
    if not {"date", "fold"} <= set(preds.columns):
        raise SystemExit("backtest() must return 'date' and 'fold' columns in preds.")

    daily = wape_by_segment(preds, segs)
    weekly = wape_by_segment(to_weekly(preds), segs)
    models = [c for c in daily.columns if c not in ("volume_share", "n_skus")]
    out = pd.concat({"daily": daily[models], "weekly": weekly[models]}, axis=1)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(out.round(3).to_string())
    Path("artifacts").mkdir(exist_ok=True)
    out.round(4).to_csv("artifacts/weekly_wape.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv")
