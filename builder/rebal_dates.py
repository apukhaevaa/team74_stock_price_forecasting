import pandas as pd
from builder.config import PATH

def index_dates(base_dict: dict) -> list:
    data = pd.read_csv(f"{PATH}/all_forecast_results.csv")
    ticker = data.groupby('Ticker')['date'].count().sort_values().index[-1]
    return base_dict[ticker].index


def rebal_dates(data: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return data[data.dayofweek == 4]