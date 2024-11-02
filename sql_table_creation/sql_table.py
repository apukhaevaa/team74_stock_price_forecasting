from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.sql import func
from config import conn_url
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Boolean,
    Date,
    Time,
    DateTime,
    Numeric,
    BigInteger,
    ForeignKey,
)


engine = create_engine(conn_url)

metadata = MetaData()

prop_funds_today_trades = Table(
    "stock_data",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("Date", DateTime),
    Column("Adj Close", Numeric(19, 6)),
    Column("Close", Numeric(19, 6)),
    Column("High", Numeric(19, 6)),
    Column("Low", Numeric(19, 6)),
    Column("Open", Numeric(19, 6)),
    Column("Volume", Numeric(19, 6)),
    mysql_engine="InnoDB",
)

def create_tradedb():
    """Create database if it doesn't exist and create all tables."""
    # if not database_exists(conn_url):
    #     create_database(conn_url)

    metadata.create_all(engine)

if __name__=="__main__":
    create_tradedb()