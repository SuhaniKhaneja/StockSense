"""REST API over the saved forecasts. Run: uvicorn api:app --reload"""
import json
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from stocksense.inventory import recommend, simulate_stock

ART = Path(__file__).parent / "artifacts"
app = FastAPI(title="StockSense API", version="1.0")


def _forecast():
    path = ART / "forecast.csv"
    if not path.exists():
        raise HTTPException(503, "No forecasts yet. Run `python -m stocksense.pipeline` first.")
    return pd.read_csv(path, parse_dates=["date"])


@app.get("/health")
def health():
    return {"status": "ok", "forecasts_ready": (ART / "forecast.csv").exists()}


@app.get("/metrics")
def metrics():
    path = ART / "backtest_summary.csv"
    if not path.exists():
        raise HTTPException(503, "No backtest results yet. Run the pipeline first.")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/forecast/{sku}")
def forecast(sku: str):
    fc = _forecast()
    f = fc[fc["sku"] == sku]
    if f.empty:
        raise HTTPException(404, f"Unknown sku '{sku}'")
    f = f.assign(date=f["date"].dt.strftime("%Y-%m-%d")).round(2)
    return {"sku": sku, "forecast": f[["date", "pred", "lo", "hi"]].to_dict(orient="records")}


@app.get("/recommendations")
def recommendations(lead_time: int = Query(7, ge=1, le=14), review: int = Query(7, ge=1, le=14),
                    service: float = Query(0.95, gt=0.5, lt=0.999)):
    fc = _forecast()
    stock = simulate_stock(sorted(fc["sku"].unique()), fc)
    return recommend(fc, stock, lead_time, review, service).to_dict(orient="records")
