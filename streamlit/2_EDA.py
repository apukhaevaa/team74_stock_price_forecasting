import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
from streamlit_lightweight_charts import renderLightweightCharts
import plotly.graph_objects as go

from custom_logger import get_custom_logger
logger = get_custom_logger("ANY_PAGE_LOGGER")
logger.debug("Page logger initialized")

st.set_page_config(layout="wide")
st.title("Exploratory Data Analysis (Daily Bars)")

def get_all_tickers():
    url = "http://127.0.0.1:8000/daily_bars/tickers"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        if data["status"] == "ok":
            return data["tickers"]
        else:
            return []
    else:
        st.error(f"Error fetching tickers: {resp.status_code}")
        return []

def get_daily_bars_data(ticker):
    url = f"http://127.0.0.1:8000/daily_bars/data?ticker={ticker}"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        if data["status"] == "ok":
            return data["data"]
        else:
            st.warning(data.get("message", "No data?"))
            return []
    else:
        st.error(f"Error fetching data for ticker={ticker}: {resp.status_code}")
        return []

tickers_list = get_all_tickers()
if not tickers_list:
    st.error("No tickers from server or server error.")
    st.stop()

ticker = st.sidebar.selectbox("Choose ticker", tickers_list)
page_width = st.sidebar.slider("Chart Width (px)", min_value=600, max_value=2200, value=1200)

if ticker:
    data_list = get_daily_bars_data(ticker)
    if not data_list:
        st.error(f"No data for {ticker}")
        st.stop()

    data = pd.DataFrame(data_list)
    data["time"] = pd.to_datetime(data["time"], format="%Y-%m-%d")
    data.sort_values("time", inplace=True)
    data.set_index("time", inplace=True)

    st.subheader(ticker)

    timeframe = st.selectbox("Choose timeframe", ["daily", "weekly", "monthly", "yearly"])
    reset_dict = {"daily": "1D", "weekly": "W", "monthly": "M", "yearly": "Y"}
    freq = reset_dict.get(timeframe, "1D")

    resampled_data = (
        data.resample(freq)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .dropna()
        .reset_index()
    )

    if resampled_data.empty:
        st.error("No data available for this timeframe.")
    else:
        COLOR_BULL = 'rgba(38,166,154,0.9)'
        COLOR_BEAR = 'rgba(239,83,80,0.9)'
        resampled_data["color"] = np.where(
            resampled_data["open"] > resampled_data["close"],
            COLOR_BEAR,
            COLOR_BULL
        )

        resampled_data["time"] = resampled_data["time"].dt.strftime("%Y-%m-%d")

        candles = json.loads(resampled_data.to_json(orient="records"))
        volume_data = resampled_data.rename(columns={"volume": "value"})
        volume = json.loads(volume_data.to_json(orient="records"))

        chartMultipaneOptions = [
            {
                "width": page_width,
                "height": 600,
                "layout": {
                    "background": {"type": "solid", "color": 'white'},
                    "textColor": "black"
                },
                "grid": {
                    "vertLines": {"color": "rgba(197, 203, 206, 0.5)"},
                    "horzLines": {"color": "rgba(197, 203, 206, 0.5)"}
                },
                "crosshair": {"mode": 0},
                "priceScale": {"borderColor": "rgba(197, 203, 206, 0.8)"},
                "timeScale": {
                    "borderColor": "rgba(197, 203, 206, 0.8)",
                    "barSpacing": 15
                },
                "watermark": {
                    "visible": True,
                    "fontSize": 48,
                    "horzAlign": 'center',
                    "vertAlign": 'center',
                    "color": 'rgba(171, 71, 188, 0.3)',
                    "text": f"{ticker} - {timeframe.upper()}",
                }
            },
            {
                "width": page_width,
                "height": 300,
                "layout": {
                    "background": {"type": 'solid', "color": 'transparent'},
                    "textColor": 'black',
                },
                "grid": {
                    "vertLines": {"color": 'rgba(42, 46, 57, 0)'},
                    "horzLines": {"color": 'rgba(42, 46, 57, 0.6)'}
                },
                "timeScale": {"visible": False},
                "watermark": {
                    "visible": True,
                    "fontSize": 18,
                    "horzAlign": 'left',
                    "vertAlign": 'top',
                    "color": 'rgba(171, 71, 188, 0.7)',
                    "text": 'Volume',
                }
            }
        ]

        seriesCandlestickChart = [
            {
                "type": 'Candlestick',
                "data": candles,
                "options": {
                    "upColor": COLOR_BULL,
                    "downColor": COLOR_BEAR,
                    "borderVisible": False,
                    "wickUpColor": COLOR_BULL,
                    "wickDownColor": COLOR_BEAR
                }
            }
        ]

        seriesVolumeChart = [
            {
                "type": 'Histogram',
                "data": volume,
                "options": {
                    "priceFormat": {"type": 'volume'},
                    "priceScaleId": ''
                },
                "priceScale": {
                    "scaleMargins": {"top": 0, "bottom": 0},
                    "alignLabels": False
                }
            }
        ]

        renderLightweightCharts([
            {"chart": chartMultipaneOptions[0], "series": seriesCandlestickChart},
            {"chart": chartMultipaneOptions[1], "series": seriesVolumeChart},
        ], key="multipane")

        col1, col2 = st.columns(2)

        with col1:
            st.header("first rows")
            df_for_display = resampled_data.drop(columns=["color"], errors="ignore")
            st.dataframe(df_for_display.head(8))

        with col2:
            st.header("basic stats")
            stats_df = df_for_display.describe()
            st.dataframe(stats_df)

        box_plot = go.Figure(go.Box(y=resampled_data["close"], name="Box Plot"))
        box_plot.update_layout(
            title="Box Plot based on close",
            yaxis_title="Values"
        )
        st.plotly_chart(box_plot)
