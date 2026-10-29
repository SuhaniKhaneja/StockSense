"""Demand-pattern segmentation (Syntetos-Boylan) and per-segment forecast error.

Usage:  python -m stocksense.segments .\\data\\sales.csv
Writes: artifacts/segment_wape.csv and artifacts/segments.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ADI_CUTOFF = 1.32   # average demand interval
CV2_CUTOFF = 0.49   # squared coefficient of variation of non-zero demand sizes


def _label(adi, cv2):
    if adi < ADI_CUTOFF:
        return "smooth" if cv2 < CV2_CUTOFF else "erratic"
    return "intermittent" if cv2 < CV2_CUTOFF else "lumpy"


def classify_skus(df, until=None):
    """One row per SKU: adi, cv2, segment. Pass `until` to classify using only
    history that is known before the earliest backtest fold (avoids leakage)."""
    d = df if until is None else df[df["date"] <= until]
    rows = []
    for sku, g in d.groupby("sku"):
        u = g["units"].to_numpy(dtype=float)
        nz = u[u > 0]
        if len(nz) == 0:
            continue
        adi = len(u) / len(nz)
        cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 else 0.0
        rows.append({"sku": sku, "adi": adi, "cv2": cv2, "segment": _label(adi, cv2)})
    return pd.DataFrame(rows)


def _wape(y, p):
    return float(np.abs(y - p).sum() / y.sum()) if y.sum() > 0 else np.nan


def wape_by_segment(preds, segs):
    """preds: columns model, sku, y, p (as returned by backtest). Pooled over folds."""
    m = preds.merge(segs[["sku", "segment"]], on="sku", how="inner")
    out = []
    for (model, seg), g in m.groupby(["model", "segment"]):
        out.append({"model": model, "segment": seg, "wape": _wape(g["y"], g["p"])})
    for model, g in m.groupby("model"):
        out.append({"model": model, "segment": "ALL", "wape": _wape(g["y"], g["p"])})
    table = pd.DataFrame(out).pivot(index="segment", columns="model", values="wape")

    ref = m[m["model"] == m["model"].iloc[0]]
    info = ref.groupby("segment").agg(units=("y", "sum"))
    info["volume_share"] = info["units"] / info["units"].sum()
    n_sku = segs.groupby("segment")["sku"].nunique()
    info["n_skus"] = n_sku
    info.loc["ALL"] = [info["units"].sum(), 1.0, int(n_sku.sum())]
    return info.drop(columns="units").join(table).sort_values("volume_share", ascending=False)


def main(path):
    from .backtest import backtest, HORIZON  # HORIZON re-exported from features

    df = pd.read_csv(path, parse_dates=["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=4 * HORIZON)  # before the earliest fold
    segs = classify_skus(df, until=cutoff)
    _, _, preds = backtest(df)
    table = wape_by_segment(preds, segs)

    pd.set_option("display.width", 140)
    print(table.round(3))
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    table.round(4).to_csv(out / "segment_wape.csv")
    segs.round(4).to_csv(out / "segments.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv")
