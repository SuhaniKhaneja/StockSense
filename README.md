# StockSense: Demand Forecasting and Reorder Recommendations

StockSense is a demand forecasting and inventory planning project that forecasts daily product demand 14 days ahead, estimates prediction intervals, and generates reorder recommendations using safety stock and reorder points.

It uses Python, pandas, scikit-learn, Streamlit, FastAPI, and SQLite. The project supports both synthetic sales data and real-world retail transaction data.

## Key Features

- **Demand forecasting:** Predicts product-level daily demand for the next 14 days.
- **Model comparison:** Compares Gradient Boosting against seasonal forecasting baselines.
- **Rolling-origin backtesting:** Evaluates models over four 14-day forecast folds.
- **Uncertainty estimation:** Generates prediction intervals and measures interval coverage.
- **Inventory recommendations:** Calculates reorder points, safety stock, and suggested order quantities.
- **Interactive dashboard:** Explores forecasts, model performance, and inventory recommendations using Streamlit.
- **REST API:** Exposes forecasting functionality through FastAPI.
- **Statistical evaluation:** Uses bootstrap confidence intervals to assess relative WAPE reductions against baseline models.

## Results on Real Data — UCI Online Retail II

The forecasting pipeline was evaluated on a daily sales panel derived from the UCI Online Retail II dataset.

**Dataset used:** 136 products across 739 days.

The evaluation compares Gradient Boosting against two seasonal baselines using rolling-origin backtesting.

| Model | WAPE | MAE | Bias | 80% Interval Coverage |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.8887 | 39.1489 | 0.1980 | 0.9242 |
| Seasonal Average (4 weeks) | 0.9913 | 43.5893 | -0.0581 | N/A |
| Seasonal Naive | 1.1437 | 50.3990 | -0.0239 | N/A |

### Baseline Comparison

- **22.3% lower WAPE** than Seasonal Naive, with a 95% bootstrap confidence interval of 18.2%–26.5%.
- **10.2% lower WAPE** than Seasonal Average, with a 95% bootstrap confidence interval of 5.2%–14.9%.

Both comparisons were calculated across 136 products.

Gradient Boosting outperformed the selected seasonal baselines in this evaluation. However, its real-data WAPE of 0.8887 (88.87%) remains high, so these results demonstrate relative improvement over the baselines rather than high absolute forecasting accuracy.

The nominal 80% prediction interval had an observed coverage of 92.42% in this backtest. Further calibration may be needed.

## Dataset and Preprocessing

The project uses the [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii), which contains historical online retail transactions.

The preprocessing pipeline in `stocksense/realdata.py`:

- Reads transaction data from CSV or Excel files.
- Removes cancelled transactions and non-product codes.
- Aggregates transaction quantities into daily product-level sales.
- Selects products based on sales volume and minimum active days.
- Creates a daily panel, representing missing product-days as zero sales.
- Produces a standardized dataset for forecasting.

Promotional information is unavailable in this dataset, so the `promo` feature is set to zero. Treating missing product-days as zero demand is a modelling assumption and may affect evaluation results.

## Tech Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Data storage and aggregation:** SQLite
- **Dashboard:** Streamlit
- **API:** FastAPI
- **Testing:** pytest
- **Evaluation:** WAPE, MAE, bias, interval coverage, and bootstrap confidence intervals

## How It Works

### 1. Feature Engineering

Implemented in `stocksense/features.py`.

The feature pipeline includes:
- Same-weekday demand lags: 14, 21, 28, and 35 days.
- Rolling demand averages over 7 and 28 days.
- Calendar features.
- Promotional flags.
- Price-ratio features.

Historical features are designed to avoid using future demand information.

### 2. Forecasting Models

Implemented in `stocksense/models.py`.

The project compares:
- Seasonal Naive.
- Seasonal Average over four weeks.
- Gradient Boosting using scikit-learn's `HistGradientBoostingRegressor` with Poisson loss.

Additional quantile models produce prediction intervals.

### 3. Model Evaluation

Implemented in `stocksense/backtest.py`.

The pipeline uses rolling-origin backtesting with four folds and a 14-day forecast horizon. Each fold is trained using the data available at its forecast origin.

Evaluation metrics include:
- Weighted Absolute Percentage Error (WAPE).
- Mean Absolute Error (MAE).
- Forecast bias.
- Prediction-interval coverage.

Bootstrap comparisons estimate relative WAPE reductions and their 95% confidence intervals.

### 4. Inventory Recommendations

Implemented in `stocksense/inventory.py`.

The inventory logic uses forecast demand, lead time, and safety stock to estimate reorder points and suggested order quantities.

Inventory quantities and stock levels shown in the dashboard are simulated rather than live inventory data.

## Run Locally

### Requirements

- Python 3.12
- Git

### Installation

```powershell
git clone https://github.com/SuhaniKhaneja/StockSense.git
cd StockSense

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Run the Synthetic-Data Pipeline

```powershell
python -m stocksense.pipeline
```

This generates synthetic sales data, runs backtesting, and writes outputs to the `artifacts/` directory.

### Run the Dashboard

```powershell
streamlit run app.py
```

### Run the REST API

```powershell
uvicorn api:app --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

### Run the Tests

```powershell
python -m pytest -q
```

## Run on Real Data

Download the UCI Online Retail II dataset and place the Excel file in the project's `data/` directory.

Convert the Excel data into a daily sales panel:

```powershell
python -m stocksense.realdata ".\data\online_retail_II.xlsx"
```

Run the forecasting and backtesting pipeline using the generated data:

```powershell
python -m stocksense.pipeline ".\data\sales.csv"
```

The pipeline writes forecasts and evaluation outputs to `artifacts/`.

Run the dashboard:

```powershell
streamlit run app.py
```

## Synthetic-Data Benchmark

The original synthetic benchmark uses seed 42 and 30 products across three years. These results are reported separately from the real-data evaluation.

| Model | WAPE | MAE | 80% Interval Coverage |
|---|---:|---:|---:|
| Gradient Boosting | 0.282 | 11.90 | 0.78 |
| Seasonal Average (4 weeks) | 0.328 | 13.83 | N/A |
| Seasonal Naive | 0.401 | 16.93 | N/A |

Synthetic results illustrate model behaviour on generated data; they should not be treated as expected real-world accuracy.

## Project Structure

```text
StockSense/
├── artifacts/
│   ├── backtest_folds.csv
│   ├── backtest_summary.csv
│   ├── bootstrap.json
│   ├── forecast.csv
│   └── meta.json
├── data/
│   └── sales.csv
├── stocksense/
│   ├── backtest.py
│   ├── data.py
│   ├── features.py
│   ├── inventory.py
│   ├── metrics.py
│   ├── models.py
│   ├── pipeline.py
│   └── realdata.py
├── tests/
├── api.py
├── app.py
├── requirements.txt
└── README.md
```

## Limitations and Future Improvements

- Real-data WAPE remains high, so further feature engineering, model tuning, and evaluation are needed.
- The dataset does not contain promotional information; promotions are assumed to be zero.
- Missing product-days are treated as zero demand, which may not always represent true demand.
- Prediction intervals may need calibration because observed coverage exceeded the nominal 80% level.
- Inventory levels in the dashboard are simulated.
- Future prices and promotions are not modelled dynamically.
- The current implementation does not explicitly model hierarchical product relationships, holidays, or inventory cost optimization.

## Future Scope

- Improve forecasting accuracy through feature engineering and systematic hyperparameter tuning.
- Evaluate performance by product, demand pattern, and forecast horizon.
- Calibrate prediction intervals.
- Add holiday and promotional features where data is available.
- Incorporate actual inventory, supplier lead times, and stockout information.
- Optimize reorder quantities based on service-level and holding-cost objectives.

---

**Author:** [Suhani Khaneja](https://github.com/SuhaniKhaneja)

**Repository:** [github.com/SuhaniKhaneja/StockSense](https://github.com/SuhaniKhaneja/StockSense)
