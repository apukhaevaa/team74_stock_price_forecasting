conn_url = "postgresql://neondb_owner:D0WAiyRxLBz2@ep-summer-sky-a5xqmm7o.us-east-2.aws.neon.tech/stock_data?sslmode=require"

most_traded_tickers = [
    # Акции
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META", "JPM", "WMT", "NFLX",
    "BABA", "DIS", "PFE", "VZ", "KO", "INTC", "CSCO", "ADBE", "CMCSA", "T",

    # Сырьевые товары
    "CL=F", "GC=F", "SI=F", "HG=F", "ZS=F",

    # Индексы
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX",

    # Облигации
    "^TNX", "^IRX", "^TYX",

    # Валюты
    "EURUSD=X", "JPYUSD=X", "GBPUSD=X", "AUDUSD=X", "CADUSD=X",
    "CHFUSD=X", "CNYUSD=X", "SGDUSD=X", "HKDUSD=X"
]

# Группы активов для анализа
asset_classes = {
    "Equities": [
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META", "JPM", "WMT", "NFLX",
        "BABA", "DIS", "PFE", "VZ", "KO", "INTC", "CSCO", "ADBE", "CMCSA", "T"
    ],
    "Commodities": ["CL=F", "GC=F", "SI=F", "HG=F", "ZS=F"],
    "Indices": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
    "Bond Indices": ["^TNX", "^IRX", "^TYX"],
    "Currencies": [
        "EURUSD=X", "JPYUSD=X", "GBPUSD=X", "AUDUSD=X", "CADUSD=X",
        "CHFUSD=X", "CNYUSD=X", "SGDUSD=X", "HKDUSD=X"
    ]
}

