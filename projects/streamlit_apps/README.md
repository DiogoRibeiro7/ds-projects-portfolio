# Streamlit Apps

Interactive dashboards for experimentation and forecasting.

## Apps
- **A/B Test Calculator**: sample size + power planning.
- **Sales Forecast Explorer**: visualize synthetic series and ARIMA forecasts.

## Run
```bash
streamlit run ab_test_calculator.py
streamlit run sales_forecast_explorer.py
```

## Deployment checklist

Before promoting a dashboard:

1. Pin runtime dependencies in `requirements.txt` and validate imports with a clean install.
2. Add a basic smoke test (at least one `streamlit run <app> --server.headless true` command) to your CI or release script.
3. Verify sensitive values are loaded from environment variables, never hardcoded.
4. Confirm static assets and expected routes render in a fresh container.
5. Capture a short manual QA checklist (startup, filters, charts, error handling) after each release.

Example headless smoke run:

```bash
streamlit run ab_test_calculator.py --server.headless true
```
