import streamlit as st
import requests
import pandas as pd

from custom_logger import get_custom_logger

logger = get_custom_logger("ANY_PAGE_LOGGER")
logger.debug("Page logger initialized")

BACKEND_URL = "http://localhost:8000"

st.title("User Data: Forecast with ARIMA")
st.write("Click the button to fit an ARIMA model on user_data.csv (if uploaded).")

st.write("ARIMA parameters (p, d, q)")
p_val = st.number_input("p", min_value=0, max_value=5, value=1)
d_val = st.number_input("d", min_value=0, max_value=2, value=1)
q_val = st.number_input("q", min_value=0, max_value=5, value=1)

if st.button("Fit user data (ARIMA)"):
    params = {"max_wait": 10, "p": p_val, "d": d_val, "q": q_val}
    with st.spinner("Fitting might take time..."):
        resp = requests.post(f"{BACKEND_URL}/fit_user_data", params=params)
    if resp.status_code == 200:
        data = resp.json()
        if data["status"] == "ok":
            st.success(
                f"Training OK. {data['processed_tickers_count']} tickers processed. "
                f"Rows={data['forecast_rows']}. {data['message']}"
            )
        elif data["status"] == "timeout":
            st.warning(f"Timeout: {data['message']}")
        else:
            st.error(data.get("detail", "Error on training."))
    else:
        st.error(f"HTTP {resp.status_code}")

st.markdown("---")
st.write("View forecast data (if the user_arima model is in the store).")

if st.button("Show forecast data"):
    params_show = {"model_id": "user_arima"}
    resp_show = requests.get(f"{BACKEND_URL}/predict", params=params_show)
    if resp_show.status_code == 200:
        data_show = resp_show.json()
        if data_show["status"] == "ok":
            forecast_rows = data_show.get("forecast", [])
            if not forecast_rows:
                st.info("No forecast rows in user_arima.")
            else:
                df_pred = pd.DataFrame(forecast_rows)
                st.dataframe(df_pred)
        else:
            st.error(data_show.get("message", "No forecast."))
    else:
        st.error(f"HTTP {resp_show.status_code}")
