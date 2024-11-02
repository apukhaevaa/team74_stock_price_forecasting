import yfinance as yf
from tqdm import tqdm
from config import most_traded_tickers
import pandas as pd
from config import conn_url
from datetime import datetime

def load_data():

    data = pd.DataFrame()

    for ticker in tqdm(most_traded_tickers):

        ticker_data = yf.download(tickers='AAPL', start='1990-01-01', interval='1d')
        ticker_data.columns = ticker_data.columns.droplevel(1)
        ticker_data = ticker_data.reset_index(drop=False)
        ticker_data.columns.name = None
        ticker_data['ticker'] = ticker
        data = pd.concat([data, ticker_data])


    data.to_sql(
        name='stock_data',
        con=conn_url,
        if_exists='replace',
        index=False

    )

    print(f"Данные обновлены в {datetime.now()}")

    return data





if __name__ == '__main__':
    print(load_data())