# StockSense: demand forecasting and reorder recommendations

Forecasts daily unit demand per product 14 days ahead, with an 80% uncertainty range,
and turns the forecast into reorder-point recommendations (safety stock, order quantity).
No API keys, databases or paid services needed.

## Run it (Windows PowerShell, Python 3.12)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m stocksense.pipeline      # generates data, backtests, writes artifacts/
streamlit run app.py
uvicorn api:app --reload          # REST API docs at http://127.0.0.1:8000/docs
python -m pytest -q                # 13 tests
```
## Run on real data (UCI Online Retail II)
Download the dataset (CSV or XLSX) from the UCI repository or its Kaggle copy and check its licence/terms first.
```powershell
python -m stocksense.realdata path\to\online_retail_II.csv --top 150   # SQL layer -> data\sales.csv
python -m stocksense.pipeline data\sales.csv                            # backtest + forecasts
streamlit run app.py
```
The SQL step (SQLite, `realdata.py`) drops cancellations and non-product codes, aggregates to daily units,
and keeps the top-N products by volume with enough active days. Days with no sales become zeros, so shop
closures count as zero demand. Promotions are not in this dataset (promo = 0).
Any other CSV with `date, sku, units` (optional `price, promo, category`) also works with the pipeline.

## How it works
- **Features** (`features.py`): same-weekday lags (14/21/28/35 days), 7- and 28-day rolling means,
  calendar terms, promo flag, price ratio. Every history feature is at least 14 days old, so
  there is no leakage; a test enforces this.
- **Models** (`models.py`): two baselines (seasonal naive, 4-week seasonal average) versus a
  gradient-boosting model (scikit-learn `HistGradientBoostingRegressor`, Poisson loss) plus
  10th/90th percentile models for the interval. One global model across all products.
- **Evaluation** (`backtest.py`): rolling-origin backtest, 4 folds x 14 days, each fold trained only
  on data available at its forecast date. Metrics: WAPE, MAE, bias, 80% interval coverage.
- **Inventory** (`inventory.py`): reorder point = lead-time demand + z * sigma * sqrt(lead time),
  sigma derived from the forecast interval. Order quantity tops stock up to cover lead time + review days.

## Results (synthetic data, seed 42, 30 products x 3 years)
| Model | WAPE | MAE | 80% coverage |
|---|---|---|---|
| Gradient boosting | 0.282 | 11.90 | 0.78 |
| Seasonal average (4 wk) | 0.328 | 13.83 | n/a |
| Seasonal naive | 0.401 | 16.93 | n/a |

## Statistical check
The pipeline also reports the relative WAPE reduction versus each baseline with a 95% bootstrap interval
(products resampled), saved to `artifacts/bootstrap.json` and shown in the app.

## Limitations
- The bundled dataset is **synthetic** (generator in `data.py`), so these numbers show the method works,
  not how it performs on real retail data. Run it on a real dataset before quoting accuracy.
- Stock levels in the UI are simulated. Future promos and prices are assumed (no promo, median price).
- No hierarchical or holiday effects; no cost optimisation of order quantities.
