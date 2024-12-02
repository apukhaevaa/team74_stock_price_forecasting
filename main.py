import pandas as pd
from sqlalchemy import create_engine
from config import conn_macro_stats, conn_macro_muls, conn_prices
from tqdm import tqdm

def get_data():

    conns = [conn_macro_stats, conn_macro_muls, conn_prices]

    for _ in tqdm(conns):

        engine = create_engine(_)

        with engine.begin() as conn:

            if _ == conn_macro_stats:
                macro_stats = pd.read_sql(
                    "select * from your_table_name",
                    conn
                )

            elif _ == conn_macro_muls:
                macro_muls = pd.read_sql(
                    "select * from your_table_name",
                    conn
                )

            elif _ == conn_prices:
                bars = pd.read_sql(
                    "select * from your_table_name",
                    conn
                )

    return macro_stats, macro_muls, bars


if __name__ == '__main__':
    macro_stats, macro_muls, bars = get_data()
    print(macro_stats)
    print(macro_muls)
    print(bars)
