# StockSense: Demand Forecasting and Inventory Evaluation on Lumpy Retail Demand

**Live demo:** https://stocksense-predictor.streamlit.app/ (free tier: may take about 30 seconds to wake up)

[![tests](https://github.com/SuhaniKhaneja/StockSense/actions/workflows/ci.yml/badge.svg)](https://github.com/SuhaniKhaneja/StockSense/actions/workflows/ci.yml)

StockSense forecasts product-level demand, estimates prediction intervals, turns forecasts into reorder recommendations, and, most importantly, **evaluates whether the forecasts actually help**. It is built on real transaction data (UCI Online Retail II) and uses leakage-safe rolling-origin backtesting with bootstrap confidence intervals.

Stack: Python, pandas, scikit-learn, Streamlit, FastAPI, SQLite, pytest.

## Key findings

On 136 products from UCI Online Retail II (739 days):

1. **Demand is lumpy.** 121 of 136 products (85% of volume) are classified as lumpy and the rest as erratic. None are smooth or intermittent. Many customers are wholesalers, so a few large orders dominate.
2. **At daily grain, gradient boosting beats naive baselines, but absolute error is high.** WAPE 0.889 vs 1.144 for seasonal naive (22.3% lower, 95% CI 18.2% to 26.5%). An error of 89% of demand means daily point forecasts are not usable on their own for this data.
3. **At weekly planning grain, simple methods win.** A 4-week average and TSB (an intermittent-demand method) are statistically tied (WAPE 0.539 and 0.522). Both clearly beat weekly naive (0.643). A gradient-boosting model trained on weekly totals is significantly worse than the 4-week average (0.639, 18.6% worse, 95% CI 10.5% to 27.7% worse).
4. **Forecast quality matters for inventory, up to a point.** In a stock simulation, a naive forecast needs roughly 10 to 13 percentage points of demand more excess stock than the 4-week average to reach the same fill rate. TSB and the 4-week average sit on essentially the same cost/service trade-off.

The honest conclusion: on this dataset a complex model does not earn its place at the weekly planning grain. The evaluation framework is what showed that.

## Dataset

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), real transactions from a UK-based online retailer between 01/12/2009 and 09/12/2011. Licensed CC BY 4.0; credit goes to the UCI Machine Learning Repository. Many of the retailer's customers are wholesalers, which is a main reason demand is lumpy.

Preprocessing (`stocksense/realdata.py`):

- Removes cancelled transactions and non-product codes.
- Aggregates transactions into daily product-level sales.
- Selects products by sales volume and minimum active days (136 products, 739 days).
- Builds a complete daily panel, with missing product-days set to zero sales. This is a modelling assumption and affects results.
- Promotion data is not available, so the `promo` feature is zero.

## Results

All results use rolling-origin backtesting: 4 folds, each forecasting 14 days from an origin, each trained only on data known at that origin. Confidence intervals are paired bootstrap over products.

### 1. Daily forecasting

| Model                      | WAPE   | MAE     | Bias    | 80% interval coverage |
|----------------------------|-------:|--------:|--------:|----------------------:|
| Gradient boosting          | 0.8887 | 39.1489 |  0.1980 | 0.9242                |
| Seasonal average (4 weeks) | 0.9913 | 43.5893 | -0.0581 | n/a                   |
| Seasonal naive             | 1.1437 | 50.3990 | -0.0239 | n/a                   |

- 22.3% lower WAPE than seasonal naive (95% CI 18.2% to 26.5%).
- 10.2% lower WAPE than the 4-week seasonal average (95% CI 5.2% to 14.9%).
- The nominal 80% interval covers 92.4% of outcomes, so the intervals are too wide (not yet calibrated).

### 2. Demand segmentation

Products are classified with the Syntetos-Boylan scheme (ADI cutoff 1.32, CV^2 cutoff 0.49), using only history from before the earliest backtest fold. Daily WAPE, pooled over folds:

| Segment | Products | Share of volume | Gradient boosting | Seasonal avg | Seasonal naive |
|---------|---------:|----------------:|------------------:|-------------:|---------------:|
| All     | 136      | 100%            | 0.883             | 0.983        | 1.136          |
| Lumpy   | 121      | 84.7%           | 0.886             | 1.005        | 1.152          |
| Erratic | 15       | 15.3%           | 0.866             | 0.860        | 1.052          |

There is no easy segment. Gradient boosting beats the seasonal average on lumpy products but not on erratic ones (0.866 vs 0.860).

### 3. Weekly planning grain

Weekly WAPE on the same test weeks. "Daily GBM summed" adds the daily model's forecasts over each week. The other rows are weekly-native models.

| Model                       | All   | Lumpy | Erratic |
|-----------------------------|------:|------:|--------:|
| TSB                         | 0.522 | 0.541 | 0.417   |
| Weekly average (4 weeks)    | 0.539 | 0.553 | 0.465   |
| Daily GBM, summed to weeks  | 0.561 | 0.557 | 0.583   |
| Weekly GBM (weekly totals)  | 0.639 | 0.638 | 0.645   |
| Weekly naive                | 0.643 | 0.656 | 0.569   |

Paired bootstrap over 136 products (WAPE reduction; positive means the first model is better):

- TSB vs 4-week average: +3.2% (95% CI -0.9% to +7.2%). **A tie.**
- TSB vs weekly naive: +18.8% (95% CI +12.9% to +24.0%).
- Weekly GBM vs 4-week average: -18.6% (95% CI -27.7% to -10.5%).
- Weekly GBM vs TSB: -22.5% (95% CI -32.9% to -13.2%).
- Weekly GBM vs weekly naive: +0.5% (95% CI -7.6% to +8.1%). A tie.

The weekly gradient-boosting model was not tuned on the backtest folds, to avoid leaking the evaluation into the model choice. The daily GBM summed to weeks was not significance-tested against the 4-week average.

### 4. Inventory backtest

Policy: weekly review, order-up-to level, lost sales, one-week lead time. For each product and week, stock is `S = max(0, forecast + q)`, where `q` is the target quantile of that method's own recent forecast errors for that product (last 26 weeks before the fold origin). Every method's forecast uses only history through two weeks earlier. Weekly GBM is excluded because its in-sample errors would understate safety stock and flatter the comparison.

| Method           | Target | Fill rate | Cycle service | Excess stock / demand |
|------------------|-------:|----------:|--------------:|----------------------:|
| TSB              | 80%    | 0.801     | 0.737         | 0.451                 |
| TSB              | 90%    | 0.881     | 0.836         | 0.753                 |
| TSB              | 95%    | 0.926     | 0.887         | 1.091                 |
| Weekly avg (4 wk)| 80%    | 0.818     | 0.740         | 0.492                 |
| Weekly avg (4 wk)| 90%    | 0.892     | 0.844         | 0.803                 |
| Weekly avg (4 wk)| 95%    | 0.936     | 0.899         | 1.146                 |
| Weekly naive     | 80%    | 0.829     | 0.738         | 0.635                 |
| Weekly naive     | 90%    | 0.897     | 0.860         | 0.935                 |
| Weekly naive     | 95%    | 0.933     | 0.909         | 1.258                 |

How to read it:

- TSB shows lower fill rate and lower excess than the 4-week average at every target. That is a different operating point on the same trade-off, not a win. Interpolating between target levels (approximate, no confidence interval), TSB and the 4-week average need about the same excess stock for the same fill rate.
- Naive needs roughly 10 to 13 percentage points of demand more excess stock than the 4-week average for the same fill rate (approximate, by interpolation).
- No method reaches its target service level. The "95%" policies deliver 92.6% to 93.6% fill, because 26 weeks of errors per product is too little to estimate a high quantile on lumpy demand.

## How it works

- `stocksense/features.py`: leakage-safe features (same-weekday lags of 14, 21, 28, 35 days, rolling means, calendar, price ratio). Every history-based feature is at least 14 days old.
- `stocksense/models.py`: seasonal naive, 4-week seasonal average, and `HistGradientBoostingRegressor` with Poisson loss plus quantile models for the 80% interval.
- `stocksense/backtest.py`: rolling-origin backtest (4 folds, 14-day horizon). Returns per-product predictions with fold and date.
- `stocksense/segments.py`: ADI/CV^2 demand classification and WAPE by segment.
- `stocksense/weekly.py`: daily vs weekly error from the same backtest.
- `stocksense/weekly_model.py`: weekly-native models (naive, 4-week average, TSB, gradient boosting) with bootstrap CIs.
- `stocksense/inventory_backtest.py`: the stock simulation above.
- `stocksense/inventory.py`: reorder points, safety stock and order quantities for the dashboard. It sizes safety stock from the model's prediction interval with a normal approximation, which the inventory backtest suggests is not well suited to lumpy demand. `stocksense/safety.py` computes empirical safety stock from each product's own recent weekly forecast errors; the dashboard uses it by default and keeps the interval method as an option.
- `app.py` (Streamlit dashboard) and `api.py` (FastAPI). Stock levels shown in the dashboard are simulated.

## Run locally

Requires Python 3.12 and Git.

```
git clone https://github.com/SuhaniKhaneja/StockSense.git
cd StockSense
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

Synthetic-data pipeline:

```
python -m stocksense.pipeline
```

## Reproduce the real-data results

The dataset is not stored in the repo. Download it from the [UCI page](https://archive.ics.uci.edu/dataset/502/online+retail+ii), put the Excel file in `data/`, then:

```
python -m stocksense.realdata ".\data\online_retail_II.xlsx"
python -m stocksense.pipeline ".\data\sales.csv"
python -m stocksense.segments ".\data\sales.csv"
python -m stocksense.weekly ".\data\sales.csv"
python -m stocksense.weekly_model ".\data\sales.csv"
python -m stocksense.inventory_backtest ".\data\sales.csv"
```

Outputs are written to `artifacts/`. Run the dashboard with `streamlit run app.py` and the API with `uvicorn api:app --reload`.

## Synthetic benchmark

Seed 42, 30 products, three years. Reported separately because generated data says little about real-world accuracy.

| Model                      | WAPE  | MAE   | 80% interval coverage |
|----------------------------|------:|------:|----------------------:|
| Gradient boosting          | 0.282 | 11.90 | 0.78                  |
| Seasonal average (4 weeks) | 0.328 | 13.83 | n/a                   |
| Seasonal naive             | 0.401 | 16.93 | n/a                   |

## Project structure

```
StockSense/
|-- artifacts/        results (backtest, segments, weekly, inventory CSVs)
|-- data/             generated locally, not tracked in git
|-- stocksense/
|   |-- backtest.py
|   |-- data.py
|   |-- features.py
|   |-- inventory.py
|   |-- inventory_backtest.py
|   |-- metrics.py
|   |-- models.py
|   |-- pipeline.py
|   |-- realdata.py
|   |-- segments.py
|   |-- weekly.py
|   `-- weekly_model.py
|-- tests/
|-- api.py
|-- app.py
`-- requirements.txt
```

## Limitations

- One dataset, one retailer, 136 products. Findings may not transfer to smoother demand.
- Missing product-days are treated as zero demand, and no promotion or holiday data is used.
- WAPE is dominated by large orders. It is a reasonable metric here but not the only one.
- Prediction intervals are over-wide (92.4% observed coverage for a nominal 80%) and have not been calibrated.
- Inventory simulation assumes a one-week lead time, lost sales and no ordering costs. The TSB parameters are fixed, not tuned.
- Safety stock from 26 weeks of errors undershoots high service targets.
- The dashboard's empirical safety stock extends the backtested one-week policy to 1 to 4 weeks of cover; the multi-week version has not been backtested separately.
- The equal-fill-rate comparison between methods is read off by interpolation, without confidence intervals.

## Future work

- Compute the equal-fill-rate stock comparison properly, with bootstrap CIs.
- Calibrate prediction intervals (for example with conformal prediction).
- Backtest the multi-week empirical policy that the dashboard uses.
- Test on a second dataset with smoother demand to see where gradient boosting starts to pay off.

## Credits

Data: [UCI Machine Learning Repository, Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), CC BY 4.0.

Author: [Suhani Khaneja](https://github.com/SuhaniKhaneja)
