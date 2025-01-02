import streamlit as st
import requests
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go

from custom_logger import get_custom_logger

logger = get_custom_logger("ANY_PAGE_LOGGER")
logger.debug("Page logger initialized")

BASE_URL = "http://127.0.0.1:8000"
st.set_page_config(layout="wide")
st.title("ARIMA Model Page")


def list_tickers_api():
    url = f"{BASE_URL}/list_tickers"
    resp = requests.get(url)
    return resp.json()


def fit_arima_params_api(p=1, q=1, max_wait=10):
    url = f"{BASE_URL}/fit_arima_params"
    params = {"p": p, "q": q, "max_wait": max_wait}
    resp = requests.post(url, params=params)
    return resp.json()


def get_params_api(ticker: str):
    url = f"{BASE_URL}/get_params"
    resp = requests.get(url, params={"ticker": ticker})
    return resp.json()


def get_forecast_api(ticker: str):
    url = f"{BASE_URL}/get_forecast"
    resp = requests.get(url, params={"ticker": ticker})
    return resp.json()


@st.cache_data
def get_tickers_cached():
    resp = list_tickers_api()
    if resp["status"] == "ok":
        return resp["tickers"]
    return []


tickers = get_tickers_cached()
if not tickers:
    st.error("No tickers available from backend, check /list_tickers")
    st.stop()

st.sidebar.header("ARIMA parameters")
ticker = st.sidebar.selectbox("Choose ticker", tickers)
p_params = st.sidebar.selectbox("choose p param", [1, 2, 3, 4, 5])
q_params = st.sidebar.selectbox("choose q param", [1, 2, 3, 4, 5])

if st.sidebar.button("Train ARIMA"):
    placeholder = st.text("ARIMA training ...")
    start_t = time.time()
    result = fit_arima_params_api(p_params, q_params, max_wait=10)
    elapsed = time.time() - start_t
    if result["status"] == "timeout":
        st.warning(
            "The process took more than 10 seconds, default params were used."
        )
    else:
        st.success(
            f"Training done in {round(elapsed, 1)}s with p={p_params}, q={q_params}"
        )
    placeholder.text(str(result))

params_resp = get_params_api(ticker)
if params_resp["status"] == "ok":
    st.write("ARIMA params for this ticker:", params_resp["params"])
else:
    st.warning(f"Cannot get ARIMA params: {params_resp}")

forecast_resp = get_forecast_api(ticker)
if forecast_resp["status"] != "ok":
    st.error(f"Forecast error: {forecast_resp['message']}")
else:
    rows_data = forecast_resp["rows"]
    df_forecast = pd.DataFrame(rows_data)
    df_forecast["date"] = pd.to_datetime(df_forecast["date"])
    df_forecast.set_index("date", inplace=True)

    st.subheader("Time series forecasted vs actual")
    st.dataframe(df_forecast.head(8), use_container_width=True)

    st.subheader("Descriptive stats")
    st.dataframe(df_forecast.describe(), use_container_width=True)

    st.subheader("Metrics")
    if "actual" in df_forecast.columns and "predicted" in df_forecast.columns:
        mse_val = mean_squared_error(df_forecast["actual"], df_forecast["predicted"])
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(df_forecast["actual"], df_forecast["predicted"])
        df_metrics = pd.DataFrame([[mse_val, rmse_val, r2_val]],
                                  columns=["MSE", "RMSE", "R2"]).T
        df_metrics.columns = ["Value"]
        st.dataframe(df_metrics, use_container_width=True)
    else:
        st.info("Missing columns 'actual' or 'predicted'.")

    fig = go.Figure()
    if "actual" in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=df_forecast.index,
            y=df_forecast["actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="blue")
        ))
    if "predicted" in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=df_forecast.index,
            y=df_forecast["predicted"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="orange")
        ))
    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        width=2200,
        height=700
    )
    st.plotly_chart(fig)
