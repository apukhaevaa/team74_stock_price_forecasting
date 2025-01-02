import pandas as pd

def get_stats(df: pd.DataFrame):
    wealth_index = (df.pct_change() + 1).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown_series = abs(wealth_index / previous_peaks - 1)
    max_dd = drawdown_series.max()
    max_dd = pd.Series(max_dd, name='Максимальная просадка').to_frame()
    prod_ret = (df.pct_change() + 1).prod() - 1
    prod_ret = pd.Series(prod_ret, name='Накопленная доходность').to_frame()
    median = df.pct_change().median()
    median = pd.Series(median, name='Медианная доходность').to_frame()
    agv = df.pct_change().mean()
    agv = pd.Series(agv, name='Средняя доходность').to_frame()
    stats = pd.concat([prod_ret, max_dd, median, agv], axis=1)

    return stats