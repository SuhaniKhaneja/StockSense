"""Inventory backtest: does a better weekly forecast mean fewer stockouts / less excess stock?

Policy (weekly review, order-up-to, lost sales, 1-week lead time):
  S_t = max(0, forecast_t + q_t)
  forecast_t uses only history through week t-2 (same rule for every method)
  q_t     = the `target` quantile of that method's own forecast errors (actual - forecast)
            for this product over the 26 weeks before the fold origin, floored at 0
Stock at the start of week t is S_t; demand above S_t is lost.
Evaluated on the same 4 folds x 2 weeks as the forecasting backtest.

Methods: naive (lag 2), 4-week average, TSB. (Weekly GBM is left out: its in-sample
errors would understate safety stock, which is unfair to the other methods.)

Usage:  python -m stocksense.inventory_backtest .\\data\\sales.csv
Writes: artifacts/inventory_backtest.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .weekly_model import make_weekly, N_FOLDS

METHODS = {"naive": "Weekly naive", "avg4": "Weekly avg (4 wk)", "tsb": "TSB"}
TARGETS = (0.80, 0.90, 0.95)
WINDOW = 26


def tsb_forecasts(y, alpha=0.2, beta=0.1):
    """f[t] = TSB rate computed from y[:t-1] only (history through t-2); NaN for t < 2."""
    n = len(y)
    state = np.zeros(n)
    p, z = 0.5, None
    for i, v in enumerate(y):
        if v > 0:
            z = v if z is None else z + alpha * (v - z)
            p += beta * (1 - p)
        else:
            p *= (1 - beta)
        state[i] = p * z if z is not None else 0.0
    f = np.full(n, np.nan)
    if n > 2:
        f[2:] = state[:-2]
    return f


def add_forecasts(w):
    w = w.copy()
    w["f_naive"] = w["lag_2"]
    w["f_avg4"] = w["roll_4"]
    w["f_tsb"] = w.groupby("sku")["units"].transform(
        lambda s: pd.Series(tsb_forecasts(s.to_numpy(dtype=float)), index=s.index))
    return w


def simulate(w, window=WINDOW, targets=TARGETS):
    rows = []
    for k in range(N_FOLDS):
        test = w[w["week_id"].isin([2 * k, 2 * k + 1])]
        hist = w[(w["week_id"] >= 2 * k + 2) & (w["week_id"] < 2 * k + 2 + window)]
        for key, name in METHODS.items():
            col = f"f_{key}"
            h = hist.dropna(subset=[col])
            err = (h["units"] - h[col])
            for tgt in targets:
                q = err.groupby(h["sku"]).quantile(tgt).clip(lower=0)
                S = (test[col] + test["sku"].map(q).fillna(0.0)).clip(lower=0).to_numpy()
                y = test["units"].to_numpy(dtype=float)
                rows.append(pd.DataFrame({
                    "method": name, "target": tgt, "fold": k, "sku": test["sku"].to_numpy(),
                    "y": y, "S": S, "unmet": np.maximum(y - S, 0), "excess": np.maximum(S - y, 0),
                    "stockout": (y > S) & (y > 0), "demand_week": y > 0}))
    return pd.concat(rows, ignore_index=True)


def summarize(sim):
    out = []
    for (m, t), g in sim.groupby(["method", "target"]):
        out.append({"method": m, "target": t,
                    "fill_rate": 1 - g["unmet"].sum() / g["y"].sum(),
                    "cycle_service": 1 - g["stockout"].sum() / g["demand_week"].sum(),
                    "excess_ratio": g["excess"].sum() / g["y"].sum(),
                    "stock_ratio": g["S"].sum() / g["y"].sum()})
    return pd.DataFrame(out)


def bootstrap_delta(sim, model, base, target, n=2000, seed=0):
    """Paired bootstrap over SKUs. Returns dict with (point, lo, hi) for
    d_fill   = fill_rate(model) - fill_rate(base)          (positive = model better)
    d_excess = excess_ratio(base) - excess_ratio(model)    (positive = model holds less excess)"""
    s = sim[sim["target"] == target]
    a = s[s["method"] == model].groupby("sku")[["y", "unmet", "excess"]].sum()
    b = s[s["method"] == base].groupby("sku")[["y", "unmet", "excess"]].sum().reindex(a.index)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    y = a["y"].to_numpy()[idx].sum(axis=1)

    def ci(point, draws):
        return (point, *np.percentile(draws, [2.5, 97.5]))

    ya = a["y"].sum()
    d_fill_pt = (b["unmet"].sum() - a["unmet"].sum()) / ya
    d_fill = (b["unmet"].to_numpy()[idx].sum(axis=1) - a["unmet"].to_numpy()[idx].sum(axis=1)) / y
    d_exc_pt = (b["excess"].sum() - a["excess"].sum()) / ya
    d_exc = (b["excess"].to_numpy()[idx].sum(axis=1) - a["excess"].to_numpy()[idx].sum(axis=1)) / y
    return {"d_fill": ci(d_fill_pt, d_fill), "d_excess": ci(d_exc_pt, d_exc)}


def main(path):
    df = pd.read_csv(path, parse_dates=["date"])
    w = add_forecasts(make_weekly(df))
    sim = simulate(w)
    table = summarize(sim)

    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print()
    for base in ("Weekly avg (4 wk)", "Weekly naive"):
        for tgt in TARGETS:
            r = bootstrap_delta(sim, "TSB", base, tgt)
            f, e = r["d_fill"], r["d_excess"]
            print(f"TSB vs {base} @ target {tgt:.0%}: fill rate {f[0]:+.3f} (CI {f[1]:+.3f} to {f[2]:+.3f}), "
                  f"excess stock {e[0]:+.3f} of demand (CI {e[1]:+.3f} to {e[2]:+.3f})")
    Path("artifacts").mkdir(exist_ok=True)
    table.round(4).to_csv("artifacts/inventory_backtest.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv")
