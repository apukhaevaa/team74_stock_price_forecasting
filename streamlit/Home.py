import streamlit as st
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import date

st.set_page_config(layout="wide", page_title="Welcome", page_icon="📈")

@st.cache_resource
def get_custom_logger(logger_name: str = "StreamlitHomeLogger"):
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", f"{date.today():%Y-%m-%d}.log")
        file_handler = RotatingFileHandler(
            log_path,
            mode="a",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8"
        )
        fmt_str = (
            "%(asctime)s [%(levelname)-5s]: %(message)s "
            "[%(name)s — %(funcName)s:%(lineno)d]"
        )
        formatter = logging.Formatter(fmt_str)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

logger = get_custom_logger("HOME_PAGE_LOGGER")
logger.debug("Home page logger initialized")

st.title("Welcome to the Brazilian Stock Forecasting App")
logger.info("Rendering Home page")

st.markdown(
    """
We chose to focus on the Brazilian stock market due to its **high volatility**, 
which offers a wide range of price fluctuations and potentially significant 
opportunities for returns. Moreover, Brazil is an emerging market with strong 
growth potential, making it an attractive target for both domestic and 
international investors.

This application provides:
- **Asset Forecasting** based on time-series models
- **Portfolio Analysis** to help compare and track performance
- **User-Friendly Visualizations** to see trends and assess risks
- **Metrics and Indicators** for deeper insights
"""
)

logger.debug("Home page content rendered successfully")
