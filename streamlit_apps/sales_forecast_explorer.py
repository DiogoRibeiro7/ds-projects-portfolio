import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA


st.set_page_config(page_title="Sales Forecast Explorer", layout="wide")

st.title("Sales Forecast Explorer")

st.markdown("Explore a synthetic daily sales series and compare baseline vs. ARIMA forecasts.")

with st.sidebar:
    n_days = st.slider("History (days)", 180, 1095, 730, 30)
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, 1)
    noise = st.slider("Noise level", 1.0, 10.0, 4.0, 0.5)
    arima_order = st.selectbox("ARIMA order", [(2, 1, 2), (1, 1, 1), (3, 1, 1)])


def make_series(days: int, noise_scale: float, random_state: int = 42) -> pd.Series:
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    trend = np.linspace(50, 80, days)
    weekly = 8 * np.sin(2 * np.pi * np.arange(days) / 7)
    yearly = 15 * np.sin(2 * np.pi * np.arange(days) / 365)
    noise = rng.normal(0, noise_scale, days)
    values = trend + weekly + yearly + noise
    return pd.Series(values, index=dates, name="sales")


def forecast_naive(train: pd.Series, horizon: int) -> pd.Series:
    last_value = train.iloc[-1]
    forecast_index = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.Series([last_value] * horizon, index=forecast_index, name="naive")


def forecast_arima(train: pd.Series, horizon: int, order: tuple[int, int, int]) -> pd.Series:
    model = ARIMA(train, order=order)
    fitted = model.fit()
    return fitted.forecast(steps=horizon)


def main() -> None:
    series = make_series(n_days, noise)
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    naive_forecast = forecast_naive(train, horizon)
    arima_forecast = forecast_arima(train, horizon, arima_order)

    chart_df = pd.DataFrame(
        {
            "actual": test,
            "naive": naive_forecast,
            "arima": arima_forecast,
        }
    )

    st.subheader("Forecast Comparison")
    st.line_chart(chart_df)

    mae_naive = np.mean(np.abs(test.values - naive_forecast.values))
    mae_arima = np.mean(np.abs(test.values - arima_forecast.values))

    col1, col2 = st.columns(2)
    col1.metric("Naive MAE", f"{mae_naive:.2f}")
    col2.metric("ARIMA MAE", f"{mae_arima:.2f}")

    st.subheader("Full Series")
    st.line_chart(series)


if __name__ == "__main__":
    main()
