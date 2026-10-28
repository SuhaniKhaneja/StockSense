"""Run end to end: data -> backtest -> final model -> forecasts. `python -m stocksense.pipeline [sales.csv]`"""
import json
from pathlib import Path
from .data import generate_sales, load_sales
from .features import FEATURES, build_features, extend_future, HORIZON
from .backtest import backtest
from .metrics import bootstrap_wape_reduction
from .models import GBMForecaster

ROOT = Path(__file__).resolve().parent.parent


def run(csv=None, out=ROOT / "artifacts", data_path=ROOT / "data" / "sales.csv"):
    out.mkdir(parents=True, exist_ok=True)
    df = load_sales(csv) if csv else generate_sales()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)  # the app reads history from here
    folds, summary, preds = backtest(df)
    folds.to_csv(out / "backtest_folds.csv", index=False)
    summary.to_csv(out / "backtest_summary.csv")
    boot = {b: bootstrap_wape_reduction(preds, "Gradient boosting", b)
            for b in ("Seasonal naive", "Seasonal average (4 wk)")}
    with open(out / "bootstrap.json", "w") as f:
        json.dump(boot, f, indent=2)
    feats = build_features(extend_future(df))
    train = feats.dropna(subset=FEATURES + ["units"])
    model = GBMForecaster().fit(train, train["units"])
    fut = feats[feats["units"].isna()].copy()
    fut["pred"], fut["lo"], fut["hi"] = model.predict_interval(fut)
    fut[["date", "sku", "pred", "lo", "hi"]].to_csv(out / "forecast.csv", index=False)
    with open(out / "meta.json", "w") as f:
        json.dump({"horizon": HORIZON, "n_skus": int(df["sku"].nunique()), "rows": len(df),
                   "data_source": str(csv) if csv else "synthetic (seed=42)"}, f)
    return summary, boot


if __name__ == "__main__":
    import sys
    summary, boot = run(sys.argv[1] if len(sys.argv) > 1 else None)
    print(summary.round(4).to_string())
    for name, b in boot.items():
        print(f"vs {name}: WAPE {b['relative_wape_reduction']:.1%} lower "
              f"(95% CI {b['ci95'][0]:.1%} to {b['ci95'][1]:.1%}, {b['n_skus']} products)")
