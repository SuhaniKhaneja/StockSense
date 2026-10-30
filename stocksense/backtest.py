"""Rolling-origin backtest: each fold trains only on data known at its forecast origin."""
import pandas as pd
from .features import HORIZON, FEATURES, build_features
from .metrics import wape, mae, bias, coverage
from .models import SeasonalNaive, SeasonalAverage, GBMForecaster


def backtest(df, n_folds=4, horizon=HORIZON):
    feats = build_features(df, horizon).dropna(subset=FEATURES + ["units"])
    end = feats["date"].max()
    rows, preds = [], []
    for k in range(n_folds):
        test_end = end - pd.Timedelta(days=horizon * k)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        origin = test_start - pd.Timedelta(days=1)
        train = feats[feats["date"] <= origin]
        test = feats[(feats["date"] >= test_start) & (feats["date"] <= test_end)]
        for model in (SeasonalNaive(), SeasonalAverage(), GBMForecaster()):
            model.fit(train, train["units"])
            if isinstance(model, GBMForecaster):
                p, lo, hi = model.predict_interval(test)
                cov = coverage(test["units"], lo, hi)
            else:
                p, cov = model.predict(test), None
                lo = hi = float("nan")
            preds.append(pd.DataFrame({"model": model.name, "fold": k,
                                       "date": test["date"].values,
                                       "sku": test["sku"].values,
                                       "y": test["units"].values, "p": p,
                                       "lo": lo, "hi": hi}))
            rows.append({"fold": k, "model": model.name, "wape": wape(test["units"], p),
                         "mae": mae(test["units"], p), "bias": bias(test["units"], p),
                         "coverage_80": cov})
    res = pd.DataFrame(rows)
    summary = res.groupby("model").agg(wape=("wape", "mean"), mae=("mae", "mean"),
                                       bias=("bias", "mean"), coverage_80=("coverage_80", "mean"))
    return res, summary.sort_values("wape"), pd.concat(preds, ignore_index=True)
