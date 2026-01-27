# Sales Forecasting (ARIMA + Baseline)

Lightweight time-series forecasting using synthetic daily sales data.

## What it covers
- Synthetic seasonal data generation
- Train/test split with rolling forecast
- Baseline (naive) vs. ARIMA
- Metrics: MAE, RMSE, MAPE

## Run
```bash
python projects/time_series/sales_forecasting/train.py
```

## Output
- Prints metrics for baseline and ARIMA, plus sample forecasts.
