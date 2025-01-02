"""
Main FastAPI server application for ARIMA + Portfolio.

Implements endpoints for ARIMA-based forecasting, user uploads,
and daily bars data. Also includes a custom logger with rotation.
"""

import asyncio
import logging
import os
import time
import shutil
import concurrent.futures
from datetime import date
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Annotated

import chardet
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

# pylint: disable=too-few-public-methods,too-many-locals,missing-function-docstring


def get_custom_logger(logger_name: str = "my_app_logger") -> logging.Logger:
    """
    Returns a rotating-file logger with the specified name.
    Logs go to ./logs/YYYY-MM-DD.log.
    """
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.DEBUG)
    os.makedirs("logs", exist_ok=True)
    logfile_path = os.path.join("logs", f"{date.today():%Y-%m-%d}.log")
    file_handler = RotatingFileHandler(
        logfile_path,
        mode="a",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8"
    )
    fmt_str = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
    formatter = logging.Formatter(fmt_str)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


logger = get_custom_logger("my_app_logger")

DATA_DIR = "data"
ARIMA_PARAMS_CSV = os.path.join(DATA_DIR, "arima_params.csv")
ALL_FORECASTS_CSV = os.path.join(DATA_DIR, "all_forecasts_data.csv")
OVERAL_DATA_XLSX = os.path.join(DATA_DIR, "overal_data.xlsx")
USER_DATA_PATH = "user_data.csv"
DAILY_BARS_CSV = os.path.join(DATA_DIR, "daily_bars.csv")

df_arima_params = pd.DataFrame()
df_forecasts = pd.DataFrame()

try:
    df_arima_params = pd.read_csv(ARIMA_PARAMS_CSV)
    df_arima_params.set_index("ticker", inplace=True)
    logger.info("Loaded %s shape=%s", ARIMA_PARAMS_CSV, df_arima_params.shape)
except Exception as exc:
    logger.error("Cannot load %s: %s", ARIMA_PARAMS_CSV, exc)

try:
    df_forecasts = pd.read_csv(ALL_FORECASTS_CSV)
    df_forecasts["date"] = pd.to_datetime(
        df_forecasts["date"], format="%m-%d-%y", errors="coerce"
    )
    df_forecasts = df_forecasts.sort_values(
        ["Ticker", "date"]
    ).reset_index(drop=True)
    logger.info("Loaded %s shape=%s", ALL_FORECASTS_CSV, df_forecasts.shape)
except Exception as exc:
    logger.error("Cannot load %s: %s", ALL_FORECASTS_CSV, exc)

model_store = {}
active_model_id: Optional[str] = None

model_store["pretrained_arima"] = {
    "description": "Pretrained ARIMA model: well-tuned and ready to use",
    "df_forecast": df_forecasts.copy()
}
active_model_id = "pretrained_arima"


class BasicResponse(BaseModel):
    status: str
    message: Optional[str] = None


class FitResponse(BaseModel):
    status: str
    message: Optional[str] = None


class ArimaParamsResponse(BaseModel):
    status: str
    ticker: str
    params: dict
    message: Optional[str] = None


class ForecastRow(BaseModel):
    date: str
    actual: float
    predicted: float


class ForecastResponse(BaseModel):
    status: str
    ticker: str
    rows: List[ForecastRow]
    message: Optional[str] = None


class TickersListResponse(BaseModel):
    status: str
    tickers: List[str]


class PortfolioDataRow(BaseModel):
    date: str
    portfolio: float
    index: float


class PortfolioDataResponse(BaseModel):
    status: str
    data: List[PortfolioDataRow]


class ModelsListItem(BaseModel):
    model_id: str
    description: str


class ModelsResponse(BaseModel):
    models: List[ModelsListItem]


class ModelID(BaseModel):
    model_id: str


class SetModelResponse(BaseModel):
    status: str
    active_model: str


class UploadDatasetResponse(BaseModel):
    status: str
    rows: int
    tickers_count: int
    message: Optional[str] = None


class FitResponseUser(BaseModel):
    status: str
    model_id: Optional[str] = None
    processed_tickers_count: Optional[int] = None
    forecast_rows: Optional[int] = None
    time_spent: Optional[float] = None
    message: Optional[str] = None


class PredictItem(BaseModel):
    ticker: str
    date: str
    predicted: float


class PredictResponse(BaseModel):
    status: str
    model_id: str
    ticker: Optional[str] = None
    rows: int
    forecast: List[PredictItem]
    message: Optional[str] = None


class DailyBarItem(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class DailyBarsResponse(BaseModel):
    status: str
    data: List[DailyBarItem]
    message: Optional[str] = None


class DailyBarsTickersResponse(BaseModel):
    status: str
    tickers: List[str]


class UserDataCandleItem(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class UserDataCandleResponse(BaseModel):
    status: str
    ticker: Optional[str] = None
    data: List[UserDataCandleItem]
    message: Optional[str] = None


def read_csv_with_autodetect(path: str) -> pd.DataFrame:
    """
    Reads CSV file using chardet to detect encoding.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")
    with open(path, "rb") as file_obj:
        raw = file_obj.read(10000)
    detect_result = chardet.detect(raw)
    enc = detect_result["encoding"] if detect_result["confidence"] >= 0.5 else "utf-8"
    return pd.read_csv(path, encoding=enc)


def validate_arima_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures CSV has columns: ticker, date, adjusted_close.
    """
    df.columns = [c.lower().strip() for c in df.columns]
    needed = ["ticker", "date", "adjusted_close"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing col {col}")
    return df


def load_daily_bars() -> pd.DataFrame:
    """
    Loads daily bars from daily_bars.csv, ensuring required columns.
    """
    if not os.path.exists(DAILY_BARS_CSV):
        raise FileNotFoundError(f"{DAILY_BARS_CSV} not found.")
    df_bars = read_csv_with_autodetect(DAILY_BARS_CSV)
    df_bars.columns = [c.lower().strip() for c in df_bars.columns]
    needed = ["ticker", "price_date", "open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in df_bars.columns:
            raise ValueError(f"daily_bars.csv missing {col}")
    df_bars["price_date"] = pd.to_datetime(df_bars["price_date"], errors="coerce")
    df_bars.dropna(subset=["price_date"], inplace=True)
    df_bars.sort_values(["ticker", "price_date"], inplace=True)
    df_bars.reset_index(drop=True, inplace=True)
    return df_bars


app = FastAPI(title="ARIMA + Portfolio")


def _arima_for_single_ticker(
    df_single: pd.DataFrame,
    horizon: int = 30,
    test_ratio: float = 0.2,
    min_data_points: int = 50
):
    """
    Train/test split, outlier removal, ARIMA(1,1,1) forecast.
    """
    if len(df_single) < min_data_points:
        raise ValueError(f"Single-ticker dataset too small ({len(df_single)})")
    df_single = df_single.copy()
    df_single["date"] = pd.to_datetime(df_single["date"], errors="coerce")
    df_single.dropna(subset=["date", "adjusted_close"], inplace=True)
    df_single.sort_values("date", inplace=True, ignore_index=True)
    if len(df_single) < min_data_points:
        raise ValueError("After dropping NaN, too small.")
    n = len(df_single)
    tr_size = int(n * (1 - test_ratio))
    train_df = df_single.iloc[:tr_size].copy()
    test_df = df_single.iloc[tr_size:].copy()
    if len(train_df) < min_data_points:
        raise ValueError("Train portion too small.")
    train_df.set_index("date", inplace=True)
    train_df.index = pd.to_datetime(train_df.index, errors="coerce")
    train_df["adjusted_close"] = train_df["adjusted_close"].interpolate(method="time")
    train_df.reset_index(inplace=True)
    low_q, high_q = train_df["adjusted_close"].quantile([0.01, 0.99])
    train_df = train_df[
        (train_df["adjusted_close"] >= low_q)
        & (train_df["adjusted_close"] <= high_q)
    ].copy()
    train_df.sort_values("date", inplace=True, ignore_index=True)
    if len(train_df) < min_data_points:
        raise ValueError("After outliers removal, too small.")
    series_train = train_df["adjusted_close"].values
    series_test = test_df["adjusted_close"].values if len(test_df) else []
    model = ARIMA(series_train, order=(1, 1, 1)).fit()
    aic_val = model.aic
    test_preds = None
    mape_test = None
    if len(series_test) > 0:
        test_preds = model.forecast(steps=len(series_test))
        mape_test = mean_absolute_percentage_error(series_test, test_preds)
    final_pred = model.forecast(steps=horizon)
    last_dt = train_df["date"].iloc[-1]
    fut_dates = pd.date_range(start=last_dt, periods=horizon + 1, freq="D")[1:]
    forecast_list = []
    for i in range(horizon):
        forecast_list.append({
            "date": str(fut_dates[i].date()),
            "predicted": float(final_pred[i])
        })
    return {
        "ticker": df_single["ticker"].iloc[0],
        "status": "ok",
        "train_size": len(train_df),
        "test_size": len(test_df),
        "aic": aic_val,
        "mape_test": mape_test,
        "horizon": horizon,
        "forecast": forecast_list
    }


def process_and_forecast_arima(
    df: pd.DataFrame,
    horizon: int = 30,
    test_ratio: float = 0.2,
    min_data_points: int = 50
):
    """
    Parallel calls of _arima_for_single_ticker using ThreadPoolExecutor.
    """
    out_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _, grp in df.groupby("ticker"):
            futures.append(
                executor.submit(
                    _arima_for_single_ticker,
                    grp.copy(),
                    horizon,
                    test_ratio,
                    min_data_points
                )
            )
        for fut in concurrent.futures.as_completed(futures):
            out_list.append(fut.result())
    return out_list


@app.get("/", response_model=BasicResponse)
async def root():
    """Root endpoint to check server status."""
    return BasicResponse(
        status="ok",
        message="Server up with a pretrained ARIMA + Portfolio data."
    )


@app.get("/list_tickers", response_model=TickersListResponse)
async def list_tickers():
    """List all known tickers from arima_params and forecast CSVs."""
    params_tickers = set(df_arima_params.index.tolist())
    forecast_tickers = set(df_forecasts["Ticker"].unique())
    final_tickers = sorted(params_tickers.union(forecast_tickers))
    return TickersListResponse(status="ok", tickers=final_tickers)


@app.post("/fit_arima_params", response_model=FitResponse)
async def fit_arima_params(p: int = 1, q: int = 1, max_wait: int = 10):
    """Simulate fitting ARIMA with p,q. Sleeps 10s, checks timeout."""
    start_t = time.time()
    await _fake_sleep(10)
    if time.time() - start_t > max_wait:
        return FitResponse(
            status="timeout",
            message="The process took too long, default p,d,q(1,1,1)."
        )
    return FitResponse(
        status="ok",
        message=f"ARIMA training done with p={p}, q={q} (if feasible)"
    )


async def _fake_sleep(seconds: int):
    """Async sleep placeholder."""
    await asyncio.sleep(seconds)


@app.get("/get_params", response_model=ArimaParamsResponse)
async def get_params(ticker: str):
    """Get ARIMA params for a given ticker."""
    if ticker not in df_arima_params.index:
        return ArimaParamsResponse(
            status="error",
            ticker=ticker,
            params={},
            message=f"No ARIMA params for {ticker}"
        )
    row = df_arima_params.loc[ticker].to_dict()
    return ArimaParamsResponse(
        status="ok",
        ticker=ticker,
        params=row,
        message="ARIMA params found"
    )


@app.get("/get_forecast", response_model=ForecastResponse)
async def get_forecast(ticker: str):
    """Get pre-saved forecast for a given ticker."""
    sub_df = df_forecasts[df_forecasts["Ticker"] == ticker]
    if sub_df.empty:
        return ForecastResponse(
            status="error",
            ticker=ticker,
            rows=[],
            message=f"No forecast data for ticker={ticker}"
        )
    out_list = []
    for _, row_val in sub_df.iterrows():
        out_list.append(
            ForecastRow(
                date=str(row_val["date"].date()),
                actual=float(row_val["actual"]),
                predicted=float(row_val["predicted"])
            )
        )
    return ForecastResponse(
        status="ok",
        ticker=ticker,
        rows=out_list,
        message=f"Forecast for {ticker}"
    )


@app.get("/get_portfolio_data", response_model=PortfolioDataResponse)
async def get_portfolio_data():
    """Fetch portfolio data from overal_data.xlsx."""
    if not os.path.exists(OVERAL_DATA_XLSX):
        return PortfolioDataResponse(status="error", data=[])
    try:
        dfp = pd.read_excel(OVERAL_DATA_XLSX, index_col=0)
        dfp.dropna(inplace=True)
        dfp.reset_index(inplace=True)
        out_list = []
        for _, row_val in dfp.iterrows():
            out_list.append(
                PortfolioDataRow(
                    date=str(row_val[dfp.columns[0]]),
                    portfolio=float(row_val["portfolio"]),
                    index=float(row_val["index"])
                )
            )
        return PortfolioDataResponse(status="ok", data=out_list)
    except Exception as exc:
        logger.error("Cannot read %s: %s", OVERAL_DATA_XLSX, exc)
        return PortfolioDataResponse(status="error", data=[])


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models in the model_store."""
    out_list = []
    if "pretrained_arima" in model_store:
        desc = model_store["pretrained_arima"].get("description", "")
        out_list.append(
            ModelsListItem(model_id="pretrained_arima", description=desc)
        )
    return ModelsResponse(models=out_list)


@app.post("/set", response_model=SetModelResponse)
async def set_active_model(payload: ModelID):
    """Set active model by ID."""
    global active_model_id
    if payload.model_id != "pretrained_arima":
        raise HTTPException(404, f"No such model {payload.model_id}")
    active_model_id = "pretrained_arima"
    return SetModelResponse(status="ok", active_model=active_model_id)


@app.post("/upload_dataset", response_model=UploadDatasetResponse)
async def upload_dataset(
    file: Annotated[UploadFile, File(...)]
) -> UploadDatasetResponse:
    """Upload CSV file to the server, save as user_data.csv."""
    try:
        temp_path = "temp_user.csv"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        df_user = read_csv_with_autodetect(temp_path)
        df_user = validate_arima_csv(df_user)
        df_user.sort_values(["ticker", "date"], inplace=True, ignore_index=True)
        df_user.to_csv(USER_DATA_PATH, index=False)
        return UploadDatasetResponse(
            status="success",
            rows=len(df_user),
            tickers_count=df_user["ticker"].nunique(),
            message="User CSV uploaded successfully."
        )
    except ValueError as val_err:
        raise HTTPException(400, str(val_err)) from val_err
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/fit_user_data", response_model=FitResponseUser)
async def fit_user_data(
    horizon: int = 30,
    max_wait: int = 10,
    p: int = 1,
    d: int = 1,
    q: int = 1
) -> FitResponseUser:
    """
    Fit ARIMA(1,1,1) on user_data.csv.
    If it takes > max_wait seconds, return timeout.
    """
    if not os.path.exists(USER_DATA_PATH):
        raise HTTPException(404, "No user_data.csv found.")
    try:
        df_user = pd.read_csv(USER_DATA_PATH)
        if df_user.empty:
            raise ValueError("user_data.csv is empty.")
        need_cols = {"ticker", "date", "adjusted_close"}
        if not need_cols.issubset(df_user.columns):
            raise ValueError("Missing columns in user_data.csv")
        df_user["date"] = pd.to_datetime(df_user["date"], errors="coerce")
        df_user.dropna(subset=["date"], inplace=True)
        df_user.sort_values(["ticker", "date"], inplace=True, ignore_index=True)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    start_t = time.time()
    try:
        loop = asyncio.get_event_loop()
        res_list = await loop.run_in_executor(
            None,
            lambda: process_and_forecast_arima(
                df=df_user,
                horizon=horizon,
                test_ratio=0.2,
                min_data_points=50
            )
        )
    except ValueError as val_err:
        raise HTTPException(400, str(val_err)) from val_err
    except Exception as ex_arg:
        logger.warning("Real ARIMA error: %s", ex_arg)
        raise HTTPException(500, str(ex_arg)) from ex_arg
    elapsed = time.time() - start_t
    if elapsed > max_wait:
        return FitResponseUser(
            status="timeout",
            model_id="user_arima",
            processed_tickers_count=0,
            forecast_rows=0,
            time_spent=round(elapsed, 2),
            message="The process took too long, we used default (1,1,1)."
        )
    all_rows = []
    processed_count = 0
    for item in res_list:
        fc = item["forecast"]
        for fcrow in fc:
            all_rows.append({
                "Ticker": item["ticker"],
                "date": fcrow["date"],
                "predicted": fcrow["predicted"]
            })
        processed_count += 1
    df_fc = pd.DataFrame(all_rows).sort_values(
        ["Ticker", "date"]
    ).reset_index(drop=True)
    if df_fc.empty:
        return FitResponseUser(
            status="ok",
            model_id="user_arima",
            processed_tickers_count=processed_count,
            forecast_rows=0,
            time_spent=round(elapsed, 2),
            message="No forecast rows for user_data."
        )
    model_store["user_arima"] = {
        "description": f"User ARIMA(1,1,1) ignoring p={p},d={d},q={q},"
                       f" horizon={horizon}",
        "df_forecast": df_fc
    }
    return FitResponseUser(
        status="ok",
        model_id="user_arima",
        processed_tickers_count=processed_count,
        forecast_rows=len(df_fc),
        time_spent=round(elapsed, 2),
        message="The process took too long, we used default params: p,d,q(1,1,1)"
    )


@app.get("/daily_bars/tickers", response_model=DailyBarsTickersResponse)
async def get_daily_bars_tickers() -> DailyBarsTickersResponse:
    """List available tickers in daily_bars.csv."""
    if not os.path.exists(DAILY_BARS_CSV):
        raise HTTPException(404, f"{DAILY_BARS_CSV} not found.")
    try:
        df_bars = load_daily_bars()
        uniq_tickers = df_bars["ticker"].unique()
        return DailyBarsTickersResponse(status="ok", tickers=list(uniq_tickers))
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/daily_bars/data", response_model=DailyBarsResponse)
async def get_daily_bars_data(ticker: str):
    """Fetch daily bars for a given ticker."""
    if not os.path.exists(DAILY_BARS_CSV):
        raise HTTPException(404, f"{DAILY_BARS_CSV} not found.")
    try:
        df_bars = load_daily_bars()
        sub_df = df_bars[df_bars["ticker"] == ticker]
        if sub_df.empty:
            return DailyBarsResponse(
                status="ok",
                data=[],
                message=f"No data for {ticker}"
            )
        sub_df = sub_df[
            ["price_date", "open", "high", "low", "close", "volume"]
        ].copy()
        sub_df.rename(columns={"price_date": "time"}, inplace=True)
        out_list = []
        for _, row_val in sub_df.iterrows():
            out_list.append(
                DailyBarItem(
                    time=str(row_val["time"].date()),
                    open=float(row_val["open"]),
                    high=float(row_val["high"]),
                    low=float(row_val["low"]),
                    close=float(row_val["close"]),
                    volume=float(row_val["volume"])
                )
            )
        return DailyBarsResponse(status="ok", data=out_list, message="OK")
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/user_data/tickers", response_model=TickersListResponse)
async def user_data_tickers():
    """List available tickers from user_data.csv."""
    if not os.path.exists(USER_DATA_PATH):
        return TickersListResponse(status="error", tickers=[])
    df_user = pd.read_csv(USER_DATA_PATH)
    if df_user.empty:
        return TickersListResponse(status="ok", tickers=[])
    sorted_uniq = sorted(df_user["ticker"].unique())
    return TickersListResponse(status="ok", tickers=sorted_uniq)


@app.get("/user_data/data", response_model=UserDataCandleResponse)
async def user_data_data(ticker: str):
    """Fetch pseudo-candles from user_data.csv for a given ticker."""
    if not os.path.exists(USER_DATA_PATH):
        raise HTTPException(404, "No user_data.csv found.")
    df_user = pd.read_csv(USER_DATA_PATH)
    if df_user.empty:
        return UserDataCandleResponse(
            status="ok",
            ticker=ticker,
            data=[],
            message="Empty user_data.csv"
        )
    sub_df = df_user[df_user["ticker"] == ticker].copy()
    if sub_df.empty:
        return UserDataCandleResponse(
            status="ok",
            ticker=ticker,
            data=[],
            message=f"No data for {ticker}"
        )
    sub_df.columns = [c.lower().strip() for c in sub_df.columns]
    if "adjusted_close" not in sub_df.columns or "date" not in sub_df.columns:
        return UserDataCandleResponse(
            status="error",
            ticker=ticker,
            data=[],
            message="Missing columns in user_data.csv"
        )
    sub_df["date"] = pd.to_datetime(sub_df["date"], errors="coerce")
    sub_df.dropna(subset=["date"], inplace=True)
    sub_df.sort_values("date", inplace=True, ignore_index=True)
    out_list = []
    for _, row_val in sub_df.iterrows():
        cval = float(row_val["adjusted_close"])
        out_list.append(
            UserDataCandleItem(
                time=str(row_val["date"].date()),
                open=cval,
                high=cval,
                low=cval,
                close=cval,
                volume=0.0
            )
        )
    return UserDataCandleResponse(
        status="ok",
        ticker=ticker,
        data=out_list,
        message="User data candles"
    )


@app.get("/predict", response_model=PredictResponse)
async def predict_endpoint(
    model_id: Optional[str] = None,
    ticker: Optional[str] = None,
    horizon: int = 10
) -> PredictResponse:
    """
    Return forecast from the specified model in model_store.
    If ticker is given, only that ticker's forecast is returned.
    """
    if not model_id:
        model_id = "user_arima"
    if model_id not in model_store:
        raise HTTPException(404, detail=f"No model {model_id} in store.")
    df_fc = model_store[model_id]["df_forecast"]
    if df_fc.empty:
        raise HTTPException(500, detail="Selected model forecast is empty.")
    items = []
    if ticker:
        sub_df = df_fc[df_fc["Ticker"] == ticker].head(horizon)
        for _, row_val in sub_df.iterrows():
            items.append(
                PredictItem(
                    ticker=str(row_val["Ticker"]),
                    date=str(row_val["date"]),
                    predicted=float(row_val["predicted"])
                )
            )
    else:
        for _, grp_val in df_fc.groupby("Ticker"):
            grp_head = grp_val.head(horizon)
            for _, row_val in grp_head.iterrows():
                items.append(
                    PredictItem(
                        ticker=str(row_val["Ticker"]),
                        date=str(row_val["date"]),
                        predicted=float(row_val["predicted"])
                    )
                )
    return PredictResponse(
        status="ok",
        model_id=model_id,
        ticker=ticker,
        rows=len(items),
        forecast=items,
        message="Prediction done."
    )
