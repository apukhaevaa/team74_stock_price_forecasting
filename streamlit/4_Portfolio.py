import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objs as go

from custom_logger import get_custom_logger
from builder.stats import get_stats

logger = get_custom_logger("ANY_PAGE_LOGGER")
logger.debug("Page logger initialized")

BASE_URL = "http://127.0.0.1:8000"

st.title("Portfolio Builder")
st.subheader("Comparison of the forecasted portfolio with the index")


def fetch_portfolio_data():
    url = f"{BASE_URL}/get_portfolio_data"
    resp = requests.get(url)
    if resp.status_code != 200:
        st.error(f"Cannot fetch portfolio data: {resp.text}")
        return None
    jdata = resp.json()
    if jdata["status"] != "ok":
        st.error("Server returned status != ok for portfolio data")
        return None
    records = jdata["data"]
    df_temp = pd.DataFrame(records)
    if "date" in df_temp.columns:
        df_temp["date"] = pd.to_datetime(df_temp["date"], errors="coerce")
        df_temp.set_index("date", inplace=True)
    return df_temp


df_portfolio = fetch_portfolio_data()
if df_portfolio is None or df_portfolio.empty:
    st.warning("No portfolio data to display.")
    st.stop()

df_portfolio.dropna(inplace=True)
st.dataframe(df_portfolio.head(10))

cumprod = (df_portfolio.pct_change().dropna() + 1).cumprod() - 1

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=cumprod.index,
        y=cumprod["portfolio"],
        mode="lines+markers",
        name="portfolio",
        line=dict(color="blue")
    )
)
fig.add_trace(
    go.Scatter(
        x=cumprod.index,
        y=cumprod["index"],
        mode="lines+markers",
        name="index",
        line=dict(color="orange")
    )
)

fig.update_layout(
    title="Chart of cumulative returns for the portfolio and the index",
    xaxis_title="Date",
    yaxis_title="Returns",
    legend_title="Legend",
    template="plotly_dark",
    width=2200,
    height=700
)

st.plotly_chart(fig)

cumprod += 1
stats = get_stats(cumprod)

st.subheader("Descriptive Statistics")
st.dataframe(stats)
