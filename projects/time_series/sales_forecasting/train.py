import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def make_synthetic_sales(n_days: int = 730, random_state: int = 42) -> pd.Series:
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

    trend = np.linspace(50, 80, n_days)
    weekly = 8 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    yearly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = rng.normal(0, 4, n_days)

    values = trend + weekly + yearly + noise
    return pd.Series(values, index=dates, name="sales")


def train_test_split(
    series: pd.Series, test_days: int = 60
) -> tuple[pd.Series, pd.Series]:
    return series.iloc[:-test_days], series.iloc[-test_days:]


def forecast_naive(train: pd.Series, horizon: int) -> pd.Series:
    last_value = train.iloc[-1]
    forecast_index = pd.date_range(
        train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    return pd.Series([last_value] * horizon, index=forecast_index)


def forecast_arima(train: pd.Series, horizon: int) -> pd.Series:
    model = ARIMA(train, order=(2, 1, 2))
    fitted = model.fit()
    forecast = fitted.forecast(steps=horizon)
    return forecast


def calculate_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    aligned = actual.align(forecast, join="inner")
    actual = aligned[0]
    forecast = aligned[1]

    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100

    return {"mae": mae, "rmse": rmse, "mape": mape}


def main() -> None:
    series = make_synthetic_sales()
    train, test = train_test_split(series)

    naive_forecast = forecast_naive(train, horizon=len(test))
    arima_forecast = forecast_arima(train, horizon=len(test))

    naive_metrics = calculate_metrics(test, naive_forecast)
    arima_metrics = calculate_metrics(test, arima_forecast)

    print("Baseline (Naive) Metrics")
    print({k: round(v, 2) for k, v in naive_metrics.items()})
    print("\nARIMA Metrics")
    print({k: round(v, 2) for k, v in arima_metrics.items()})
    print("\nSample Forecasts")
    print(
        pd.DataFrame(
            {
                "actual": test.head(5),
                "naive": naive_forecast.head(5),
                "arima": arima_forecast.head(5),
            }
        )
    )


if __name__ == "__main__":
    main()
