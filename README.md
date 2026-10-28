StockSense: demand forecasting and reorder recommendations
StockSense forecasts daily unit demand per product 14 days ahead, provides prediction intervals, and turns forecasts into inventory reorder recommendations (safety stock and order quantity). The project includes a Streamlit dashboard, a FastAPI service, backtesting, and data preparation for the UCI Online Retail II dataset.
Run it locally
Requirements: Python 3.12 and Windows PowerShell.
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the default synthetic-data pipeline
python -m stocksense.pipeline

# Start the dashboard
streamlit run app.py

# In another terminal, start the REST API
uvicorn api:app --reload
# API documentation: http://127.0.0.1:8000/docs

# Run tests
python -m pytest -q
Run on real data: UCI Online Retail II
Download the UCI Online Retail II dataset and review its licence and terms of use. The project can process an Excel workbook or a CSV file.
python .\stocksense\realdata.py ".\data\online_retail_II.xlsx" --top 150
python -m stocksense.pipeline ".\data\sales.csv"
streamlit run app.py
The data-preparation step uses SQLite to remove cancellations and non-product codes, aggregate transactions into daily units, and select products with enough active days. Missing product-days are represented as zero sales. Promotions are not present in this dataset, so promo = 0. The pipeline also accepts other CSV files with date, sku, and units columns; price, promo, and category are optional.
Real-data backtest results
The current run used a daily panel derived from UCI Online Retail II containing 136 products across 739 days. Evaluation used a rolling-origin backtest with four folds, each covering a 14-day horizon.
Model	WAPE	MAE	Bias	80% interval coverage
Gradient boosting	0.8887	39.1489	0.1980	0.9242
Seasonal average (4 wk)	0.9913	43.5893	-0.0581	n/a
Seasonal naive	1.1437	50.3990	-0.0239	n/a


Gradient Boosting's WAPE was 22.3% lower than Seasonal Naive (95% bootstrap confidence interval: 18.2% to 26.5%) and 10.2% lower than Seasonal Average (95% bootstrap confidence interval: 5.2% to 14.9%), across 136 products.

These are relative improvements over the selected baselines, not evidence of high absolute accuracy. The real-data WAPE of 0.8887 (88.87%) is still high, so the forecasts should be treated as decision support and improved and validated further before operational use. The nominal 80% prediction interval had measured coverage of 92.42% in this run.
Synthetic-data benchmark
The repository also includes a synthetic data generator (stocksense/data.py). The original synthetic benchmark used seed 42, 30 products, and three years of generated data. Keep these figures separate from the real-data results above.
Model	WAPE	MAE	80% coverage
Gradient boosting	0.282	11.90	0.78
Seasonal average (4 wk)	0.328	13.83	n/a
Seasonal naive	0.401	16.93	n/a


How it works
- Features (stocksense/features.py): weekday lags (14, 21, 28, and 35 days), 7- and 28-day rolling means, calendar features, a promotion flag, and a price ratio. The tests check that history features do not leak future values.
- Models (stocksense/models.py): seasonal-naive and four-week seasonal-average baselines compared with a global scikit-learn gradient-boosting model; quantile models estimate prediction intervals.
- Evaluation (stocksense/backtest.py): rolling-origin evaluation over four 14-day folds, using WAPE, MAE, bias, and interval coverage.
- Inventory (stocksense/inventory.py): estimates reorder points from lead-time demand and safety stock, and suggests quantities to cover lead time plus review days.
- Interfaces: app.py provides the Streamlit dashboard, while api.py exposes the FastAPI endpoints.
Limitations
- Real-data WAPE remains high; more feature engineering, careful treatment of intermittent demand, and further validation are needed.
- The Online Retail II data does not contain promotions. Zero-filling missing product-days is a modelling choice that may not always represent true demand.
- Stock levels in the UI are simulated. Future promotions and prices are assumed rather than forecast.
- The current approach does not model hierarchical product relationships or explicit holiday effects, and it does not optimise order quantities against business costs.
- Backtest performance is historical and is not a guarantee of future performance.
