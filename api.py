
"""REST API over the saved forecasts. Run: uvicorn api:app --reload"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from stocksense.inventory import recommend, simulate_stock

ART = Path(__file__).parent / "artifacts"

app = FastAPI(title="StockSense API", version="1.0")


def _forecast():
    path = ART / "forecast.csv"

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="No forecasts yet. Run `python -m stocksense.pipeline` first.",
        )

    return pd.read_csv(path, parse_dates=["date"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "forecasts_ready": (ART / "forecast.csv").exists(),
    }


@app.get("/metrics")
def metrics():
    path = ART / "backtest_summary.csv"

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="No backtest results yet. Run the pipeline first.",
        )

    df = pd.read_csv(path)

    # Replace missing and non-finite numeric values with JSON null.
    df = df.astype(object)
    df = df.where(pd.notna(df), None)
    df = df.replace([float("inf"), float("-inf")], None)

    return df.to_dict(orient="records")


@app.get("/forecast/{sku}")
def forecast(sku: str):
    fc = _forecast()
    f = fc[fc["sku"] == sku]

    if f.empty:
        raise HTTPException(status_code=404, detail=f"Unknown sku '{sku}'")

    f = f.assign(date=f["date"].dt.strftime("%Y-%m-%d")).round(2)

    return {
        "sku": sku,
        "forecast": f[["date", "pred", "lo", "hi"]].to_dict(orient="records"),
    }


@app.get("/recommendations")
def recommendations(
    lead_time: int = Query(7, ge=1, le=14),
    review: int = Query(7, ge=1, le=14),
    service: float = Query(0.95, gt=0.5, lt=0.999),
):
    fc = _forecast()
    stock = simulate_stock(sorted(fc["sku"].unique()), fc)

    result = recommend(fc, stock, lead_time, review, service)

    # Ensure any missing or non-finite values are JSON-safe.
    result = result.astype(object)
    result = result.where(pd.notna(result), None)
    result = result.replace([float("inf"), float("-inf")], None)

    return result.to_dict(orient="records")
