"""Weekly-native forecasting: train directly on weekly totals instead of summing daily forecasts.

Compares, on exactly the same 4 folds x 2 weeks as the daily backtest:
  - Weekly naive       (total of the week 2 weeks earlier)
  - Weekly avg (4 wk)  (mean of weekly totals 2..5 weeks earlier; equals the daily
                        seasonal average summed over a week)
  - TSB                (Teunter-Syntetos-Babai intermittent-demand method, flat forecast)
  - Weekly GBM         (Poisson gradient boosting on weekly lags, trained on weekly totals)

Usage:  python -m stocksense.weekly_model .\\data\\sales.csv
Writes: artifacts/weekly_native.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

from .features import HORIZON
from .segments import classify_skus, wape_by_segment

N_FOLDS = 4
FEATURES_W = ["lag_2", "lag_3", "lag_4", "lag_5", "roll_4", "roll_8", "nz_8",
              "woy_sin", "woy_cos", "sku_code"]


def make_weekly(df):
    """Weekly panel anchored at the last date, so fold k's test window is exactly
    weeks 2k and 2k+1 counted back from the end (same days as the daily backtest)."""
    end, start = df["date"].max(), df["date"].min()
    n_days = (end - start).days + 1
    nfull = n_days // 7                      # drop the partial week at the start
    d = df[["sku", "date", "units"]].copy()
    d["week_id"] = (end - d["date"]).dt.days // 7
    w = d[d["week_id"] < nfull].groupby(["sku", "week_id"])["units"].sum()
    idx = pd.MultiIndex.from_product([sorted(d["sku"].unique()), range(nfull)],
                                     names=["sku", "week_id"])
    w = w.reindex(idx, fill_value=0).reset_index()
    w["t"] = nfull - 1 - w["week_id"]        # ascending in time
    w = w.sort_values(["sku", "t"]).reset_index(drop=True)
    w["week_end"] = end - pd.to_timedelta(7 * w["week_id"], unit="D")

    g = w.groupby("sku")["units"]
    shifted = {L: g.shift(L) for L in range(2, 10)}
    for L in (2, 3, 4, 5):
        w[f"lag_{L}"] = shifted[L]
    stack = pd.concat(shifted.values(), axis=1)
    ok = stack.notna().all(axis=1)
    w["roll_4"] = w[["lag_2", "lag_3", "lag_4", "lag_5"]].mean(axis=1, skipna=False)
    w["roll_8"] = stack.mean(axis=1, skipna=False)
    w["nz_8"] = (stack > 0).mean(axis=1).where(ok)
    woy = w["week_end"].dt.isocalendar().week.astype(float)
    w["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
    w["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)
    w["sku_code"] = w["sku"].astype("category").cat.codes
    return w


def tsb_rate(y, alpha=0.2, beta=0.1):
    """Teunter-Syntetos-Babai: smoothed demand probability x smoothed demand size."""
    y = np.asarray(y, dtype=float)
    nz = y[y > 0]
    if len(nz) == 0:
        return 0.0
    z, p = nz[0], float((y > 0).mean())
    for v in y:
        if v > 0:
            p += beta * (1 - p)
            z += alpha * (v - z)
        else:
            p *= (1 - beta)
    return p * z


def run(df):
    w = make_weekly(df)
    cat_idx = [FEATURES_W.index("sku_code")]
    out = []
    for k in range(N_FOLDS):
        test = w[w["week_id"].isin([2 * k, 2 * k + 1])]
        train_all = w[w["week_id"] >= 2 * k + 2]          # target week ends on/before origin
        train = train_all.dropna(subset=FEATURES_W)
        rates = train_all.groupby("sku")["units"].apply(lambda s: tsb_rate(s.to_numpy()))
        gbm = HGB(loss="poisson", max_iter=150, learning_rate=0.05, max_leaf_nodes=8,
                  min_samples_leaf=40, random_state=0, categorical_features=cat_idx)
        gbm.fit(train[FEATURES_W], train["units"])
        preds = {
            "Weekly naive": test["lag_2"].to_numpy(),
            "Weekly avg (4 wk)": test["roll_4"].to_numpy(),
            "TSB": test["sku"].map(rates).to_numpy(),
            "Weekly GBM": np.clip(gbm.predict(test[FEATURES_W]), 0, None),
        }
        for name, p in preds.items():
            out.append(pd.DataFrame({"model": name, "fold": k, "sku": test["sku"].to_numpy(),
                                     "y": test["units"].to_numpy(dtype=float), "p": p}))
    return pd.concat(out, ignore_index=True)


def bootstrap_reduction(preds, model, base, n=2000, seed=0):
    """Paired bootstrap over SKUs: relative WAPE reduction of `model` vs `base`
    (positive = model better). Returns (point, ci_low, ci_high)."""
    err = (preds["y"] - preds["p"]).abs().groupby([preds["sku"], preds["model"]]).sum().unstack()
    em, eb = err[model].to_numpy(), err[base].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(em), size=(n, len(em)))
    red = 1 - em[idx].sum(axis=1) / eb[idx].sum(axis=1)
    return 1 - em.sum() / eb.sum(), *np.percentile(red, [2.5, 97.5])


def main(path):
    df = pd.read_csv(path, parse_dates=["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=N_FOLDS * HORIZON)
    segs = classify_skus(df, until=cutoff)
    preds = run(df)

    table = wape_by_segment(preds, segs)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(table.round(3).to_string())
    print("\nSanity check: 'Weekly avg (4 wk)' ALL should be about 0.539 (same as weekly.py).\n")
    for base in ("Weekly avg (4 wk)", "TSB", "Weekly naive"):
        pt, lo, hi = bootstrap_reduction(preds, "Weekly GBM", base)
        print(f"Weekly GBM vs {base}: WAPE {pt:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}, "
              f"{preds['sku'].nunique()} products; positive = GBM better)")
    Path("artifacts").mkdir(exist_ok=True)
    table.round(4).to_csv("artifacts/weekly_native.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv")
