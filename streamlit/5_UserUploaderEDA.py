import streamlit as st
import requests
import pandas as pd
import json
import numpy as np
from io import StringIO
from streamlit_lightweight_charts import renderLightweightCharts
import plotly.graph_objects as go

from custom_logger import get_custom_logger

logger = get_custom_logger("ANY_PAGE_LOGGER")
logger.debug("Page logger initialized")

BACKEND_URL = "http://localhost:8000"

st.set_page_config(layout="wide")
st.title("User Data: Upload & EDA")

file_upl = st.file_uploader("Choose a CSV file", type=["csv"])
if st.button("Upload to server") and file_upl:
    files = {"file": (file_upl.name, file_upl.getvalue())}
    with st.spinner("Uploading..."):
        resp = requests.post(f"{BACKEND_URL}/upload_dataset", files=files)
    if resp.status_code == 200:
        data = resp.json()
        if data["status"] == "success":
            st.success(
                f"Uploaded OK! Rows={data['rows']}, "
                f"Tickers={data['tickers_count']}"
            )
        else:
            st.error(data.get("detail", "Unknown error"))
    else:
        st.error(f"HTTP {resp.status_code}")

st.markdown("---")


def get_user_data_tickers():
    url = f"{BACKEND_URL}/user_data/tickers"
    resp = requests.get(url)
    if resp.status_code == 200:
        js_data = resp.json()
        if js_data["status"] == "ok":
            return js_data["tickers"]
        st.warning(js_data.get("message", "No tickers?"))
        return []
    st.error(f"HTTP {resp.status_code}")
    return []


def get_user_data_pseudo_candles(ticker):
    url = f"{BACKEND_URL}/user_data/data?ticker={ticker}"
    resp = requests.get(url)
    if resp.status_code == 200:
        js_data = resp.json()
        if js_data["status"] == "ok":
            return js_data["data"]
        st.warning(js_data.get("message", "No data?"))
        return []
    st.error(f"HTTP {resp.status_code}")
    return []


st.header("Explore user_data.csv")
tickers_list = get_user_data_tickers()
if not tickers_list:
    st.info("Either no user_data.csv or it is empty. Upload CSV above.")
else:
    chosen_ticker = st.selectbox("Choose ticker from user_data.csv", tickers_list)
    page_width = st.slider("Chart Width (px)", 600, 2200, 1200)
    if chosen_ticker:
        data_list = get_user_data_pseudo_candles(chosen_ticker)
        if not data_list:
            st.error(f"No data for ticker={chosen_ticker}")
        else:
            df = pd.DataFrame(data_list)
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df.sort_values("time", inplace=True)
            df.set_index("time", inplace=True)

            timeframe = st.selectbox(
                "Choose timeframe",
                ["daily", "weekly", "monthly", "yearly"]
            )
            freq_map = {
                "daily": "1D",
                "weekly": "W",
                "monthly": "M",
                "yearly": "Y"
            }
            freq = freq_map.get(timeframe, "1D")

            res = (
                df.resample(freq)
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
            if res.empty:
                st.warning("No data after resampling.")
            else:
                st.subheader(chosen_ticker)
                COLOR_BULL = "rgba(38,166,154,0.9)"
                COLOR_BEAR = "rgba(239,83,80,0.9)"

                res["color"] = np.where(
                    res["open"] > res["close"],
                    COLOR_BEAR,
                    COLOR_BULL
                )
                res["time"] = res["time"].dt.strftime("%Y-%m-%d")

                candles = json.loads(res.to_json(orient="records"))
                volume_data = res.rename(columns={"volume": "value"})
                volume = json.loads(volume_data.to_json(orient="records"))

                chart_config = [
                    {
                        "width": page_width,
                        "height": 600,
                        "layout": {
                            "background": {"type": "solid", "color": "white"},
                            "textColor": "black"
                        },
                        "grid": {
                            "vertLines": {"color": "rgba(197,203,206,0.5)"},
                            "horzLines": {"color": "rgba(197,203,206,0.5)"}
                        },
                        "crosshair": {"mode": 0},
                        "priceScale": {"borderColor": "rgba(197,203,206,0.8)"},
                        "timeScale": {
                            "borderColor": "rgba(197,203,206,0.8)",
                            "barSpacing": 15
                        },
                        "watermark": {
                            "visible": True,
                            "fontSize": 48,
                            "horzAlign": "center",
                            "vertAlign": "center",
                            "color": "rgba(171,71,188,0.3)",
                            "text": f"{chosen_ticker} - {timeframe.upper()}"
                        }
                    },
                    {
                        "width": page_width,
                        "height": 300,
                        "layout": {
                            "background": {"type": "solid", "color": "transparent"},
                            "textColor": "black"
                        },
                        "grid": {
                            "vertLines": {"color": "rgba(42,46,57,0)"},
                            "horzLines": {"color": "rgba(42,46,57,0.6)"}
                        },
                        "timeScale": {"visible": False},
                        "watermark": {
                            "visible": True,
                            "fontSize": 18,
                            "horzAlign": "left",
                            "vertAlign": "top",
                            "color": "rgba(171,71,188,0.7)",
                            "text": "Volume"
                        }
                    }
                ]

                candle_series = [
                    {
                        "type": "Candlestick",
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
                volume_series = [
                    {
                        "type": "Histogram",
                        "data": volume,
                        "options": {
                            "priceFormat": {"type": "volume"},
                            "priceScaleId": ""
                        },
                        "priceScale": {
                            "scaleMargins": {"top": 0, "bottom": 0},
                            "alignLabels": False
                        }
                    }
                ]

                renderLightweightCharts(
                    [
                        {"chart": chart_config[0], "series": candle_series},
                        {"chart": chart_config[1], "series": volume_series},
                    ],
                    key="user_data_multipane"
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.header("first rows")
                    df_display = res.drop(columns=["color"], errors="ignore")
                    st.dataframe(df_display.head(8))

                with col2:
                    st.header("basic stats")
                    st.dataframe(df_display.describe())

                box_plot = go.Figure(go.Box(y=res["close"], name="Box Plot"))
                box_plot.update_layout(
                    title="Box Plot based on close",
                    yaxis_title="Values"
                )
                st.plotly_chart(box_plot)
