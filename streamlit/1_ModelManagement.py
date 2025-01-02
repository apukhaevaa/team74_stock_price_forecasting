import streamlit as st
import requests

from custom_logger import get_custom_logger

logger = get_custom_logger("ANY_PAGE_LOGGER")
logger.debug("Page logger initialized")

BACKEND_URL = "http://localhost:8000"

st.title("Model Management")

st.subheader("About ARIMA Model")
st.markdown(
    """
**ARIMA (AutoRegressive Integrated Moving Average)** is a classical statistical 
model used for time series forecasting. It combines three concepts:

1. **AR (Autoregressive)**: captures the dependency between an observation and 
   a certain number of its lagged values.
2. **I (Integrated)**: differencing the data (subtracting past values) 
   to make it stationary.
3. **MA (Moving Average)**: incorporates the dependency between an observation 
   and a residual error from a moving average model applied to lagged observations.

**Why ARIMA can be a good choice**:
- It is relatively simple and interpretable: the parameters (p, d, q) directly 
  correspond to the AR, integration, and MA components.
- It performs well on time series without strong seasonality (or mild 
  seasonality if you extend it to SARIMA).
- It does not require massive datasets, unlike some deep learning approaches.
"""
)

st.markdown("---")

st.subheader("List of Models")
try:
    resp = requests.get(f"{BACKEND_URL}/models")
    if resp.status_code == 200:
        data = resp.json()
        models = data.get("models", [])
    else:
        st.error(f"Error: {resp.status_code}")
        models = []
except Exception as err:
    st.error(str(err))
    models = []

if not models:
    st.warning("No models found in store.")
else:
    for model_info in models:
        mdl_id = model_info["model_id"]
        desc = model_info["description"]
        st.write(f"**{mdl_id}**: {desc}")

st.subheader("Set Active Model")
model_ids = [m["model_id"] for m in models]
chosen = st.selectbox("Choose model to activate:", model_ids)

if st.button("Activate Model"):
    try:
        resp2 = requests.post(f"{BACKEND_URL}/set", json={"model_id": chosen})
        if resp2.status_code == 200:
            data2 = resp2.json()
            if data2["status"] == "ok":
                st.success(f"Active model: {data2['active_model']}")
            else:
                st.error(data2.get("detail", "Unknown error"))
        else:
            st.error(f"HTTP {resp2.status_code}")
    except Exception as ex:
        st.error(str(ex))
