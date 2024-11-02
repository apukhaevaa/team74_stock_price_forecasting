import pandas as pd
from sqlalchemy import create_engine
from config import conn_url
conn_url = "postgresql://neondb_owner:D0WAiyRxLBz2@ep-summer-sky-a5xqmm7o.us-east-2.aws.neon.tech/stock_data?sslmode=require"
engine = create_engine(conn_url)

with engine.begin() as conn:
  data = pd.read_sql(
      "select * from stock_data",
      conn
  )

if __name__ == "main__":
    print(data)