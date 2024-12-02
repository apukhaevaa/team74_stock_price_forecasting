from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.sql import func
from config import conn_macro_stats, conn_macro_muls, conn_prices
import pandas as pd

macro_muls = pd.read_excel("C:/Users/X/Desktop/multiplicators.xlsx")
macro_stats = pd.read_excel("C:/Users/X/Desktop/Brazil_Macro.xlsx")
prices = pd.read_csv("C:/Users/X/Desktop/daily_bars.csv")


try:


    macro_muls.to_sql('your_table_name', create_engine(conn_macro_muls), if_exists='replace', index=False)
    macro_stats.to_sql('your_table_name', create_engine(conn_macro_stats), if_exists='replace', index=False)
    prices.to_sql('your_table_name', create_engine(conn_prices), if_exists='replace', index=False)
    print("Data uploaded successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
