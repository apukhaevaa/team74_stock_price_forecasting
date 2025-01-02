<img width="650" alt="Снимок экрана 2025-01-02 в 23 22 31" src="https://github.com/user-attachments/assets/8cc1da7f-bad8-4068-bc00-c86e409c1144" />



### Краткое описание 

- **builder/** — содержит логику построения портфеля, расчётов, вспомогательные модули.
- **data/** — хранит CSV и другие входные данные.  
- **streamlit/** — весь фронтенд на Streamlit:
  - `Home.py` — главная страница,
  - `pages/` — подстраницы,
  - `custom_logger.py` — логирование в Streamlit.
- **requirements.txt** — зависимости для Python (FastAPI, streamlit, pandas и т.п.).
- **main.py** — сервер.



# Описание FastAPI приложения: ARIMA + Portfolio


## Содержание
1. [Общее описание](#общее-описание)
2. [Структура проекта](#структура-проекта)
3. [Описание логгера (RotatingFileHandler)](#описание-логгера)
4. [Основные глобальные переменные](#основные-глобальные-переменные)
5. [Асинхронность и параллелизм](#асинхронность-и-параллелизм)
6. [Описание эндпоинтов](#описание-эндпоинтов)
    1. [GET `/`](#1-get-)
    2. [GET `/list_tickers`](#2-get-list_tickers)
    3. [POST `/fit_arima_params`](#3-post-fit_arima_params)
    4. [GET `/get_params`](#4-get-get_params)
    5. [GET `/get_forecast`](#5-get-get_forecast)
    6. [GET `/get_portfolio_data`](#6-get-get_portfolio_data)
    7. [GET `/models`](#7-get-models)
    8. [POST `/set`](#8-post-set)
    9. [POST `/upload_dataset`](#9-post-upload_dataset)
    10. [POST `/fit_user_data`](#10-post-fit_user_data)
    11. [GET `/daily_bars/tickers`](#11-get-daily_barstickers)
    12. [GET `/daily_bars/data`](#12-get-daily_barsdata)
    13. [GET `/user_data/tickers`](#13-get-user_datatickers)
    14. [GET `/user_data/data`](#14-get-user_datadata)
    15. [GET `/predict`](#15-get-predict)
7. [Заключение](#заключение)

---

## Общее описание

Это сервер на базе **FastAPI**, предназначенный для:
1. **Прогнозирования временных рядов** (например, цен акций) с помощью моделей ARIMA.
2. **Загрузки пользовательских данных** и обучения модели на них.
3. **Получения данных портфеля** (портфель, индекс).
4. **Работы с историческими ценами** (дневными cвечами) — **daily bars**.
5. **Управления разными моделями** (предобученная ARIMA и пользовательская).


---

## Структура проекта

- **`main.py`** .
- Папка **`data`**:
  - `arima_params.csv` — ARIMA-параметры для разных тикеров.
  - `all_forecasts_data.csv` — сохранённые прогнозы (actual/predicted) по тикерам.
  - `overal_data.xlsx` — данные портфеля.
  - `daily_bars.csv` — исторические дневные свечи.
- Файл **`user_data.csv`** (сохраняется при загрузке пользовательского датасета).
- **`model_store`** — глобальный словарь с объектами-моделями (например, `pretrained_arima`, `user_arima`).

---

## Описание логгера

В начале приложения создаётся логгер через функцию:
```python
def get_custom_logger(logger_name: str = "my_app_logger") -> logging.Logger:
    ...
```
- Логи сохраняются в папке `./logs`, имя файла — текущая дата (`YYYY-MM-DD.log`).
- Логгер использует `RotatingFileHandler`, который:
  - Ограничивает размер файла.
  - Хранит не более трёх резервных копий лог-файла.
- Формат лога:
  ```
  %(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s
  ```

---

## Основные глобальные переменные

- **`DATA_DIR`** (`data`) — путь к папке с исходными данными.
- **`ARIMA_PARAMS_CSV`, `ALL_FORECASTS_CSV`, `OVERAL_DATA_XLSX`, `USER_DATA_PATH`, `DAILY_BARS_CSV`** — пути к конкретным CSV/XLSX.
- **`df_arima_params`** и **`df_forecasts`** — глобальные датафреймы, загружаемые из CSV при старте.
- **`model_store`** — в памяти хранит модели, напр. `"pretrained_arima"`, `"user_arima"`.
- **`active_model_id`** — идентификатор текущей активной модели.

При запуске в `df_arima_params` и `df_forecasts` загружаются данные из соответствующих CSV. В случае ошибки чтения пишется лог с ошибкой. Затем в `model_store` добавляется **`"pretrained_arima"`** (предобученная модель), и `active_model_id` указывает на неё.

---

## Асинхронность и параллелизм

### Асинхронность (async / await)
Приложение использует **FastAPI** с асинхронными эндпоинтами (`async def ...`).  
Также есть функция `_fake_sleep`, которая вызывает:
```python
await asyncio.sleep(seconds)
```
Это позволяет «приостановить» текущую задачу, не блокируя event loop.

### Параллелизм (ThreadPoolExecutor)
Для тяжёлых вычислений используется:
```python
with concurrent.futures.ThreadPoolExecutor() as executor:
    ...
```
и в эндпоинтах (например, `/fit_user_data`) мы делаем:
```python
loop = asyncio.get_event_loop()
res_list = await loop.run_in_executor(
    None,
    lambda: process_and_forecast_arima(...)
)
```
Таким образом, расчёты выполняются в других потоках, а сервер продолжает асинхронно обрабатывать другие запросы.

---

## Описание эндпоинтов

### 1. **GET** `/`
**Описание**  
- Корневой эндпоинт для быстрой проверки, что сервер запущен.

**Пример ответа**  
```json
{
  "status": "ok",
  "message": "Server up with a pretrained ARIMA + Portfolio data."
}
```

---

### 2. **GET** `/list_tickers`

**Описание**  
- Возвращает список всех тикеров, которые есть либо в `df_arima_params`, либо в `df_forecasts`.

**Пример ответа**  
```json
{
  "status": "ok",
  "tickers": ["AAPL", "MSFT", "TSLA", ...]
}
```

---

### 3. **POST** `/fit_arima_params`

**Параметры** (query/body):
- `p` (int) по умолчанию `1`
- `q` (int) по умолчанию `1`
- `max_wait` (int) по умолчанию `10`

**Описание**  
- Делает «фиктивное» (симуляция) обучение ARIMA с параметрами `p, q`.
- Спит (`await asyncio.sleep(10)`). Если прошло больше `max_wait`, возвращает `"timeout"`.

**Пример запроса**  
```http
POST /fit_arima_params?p=2&q=2&max_wait=5
```
**Пример ответа**  
```json
{
  "status": "ok",
  "message": "ARIMA training done with p=2, q=2 (if feasible)"
}
```
или
```json
{
  "status": "timeout",
  "message": "The process took too long, default p,d,q(1,1,1)."
}
```

---

### 4. **GET** `/get_params`

**Параметры** (query):
- `ticker: str`

**Описание**  
- Возвращает ARIMA-параметры (из `df_arima_params`) для указанного тикера.

**Пример запроса**  
```http
GET /get_params?ticker=AAPL
```

**Пример ответа**  
```json
{
  "status": "ok",
  "ticker": "AAPL",
  "params": {
    "p": 1,
    "d": 1,
    "q": 1
  },
  "message": "ARIMA params found"
}
```
Если нет данных по тикеру:
```json
{
  "status": "error",
  "ticker": "UNKNOWN",
  "params": {},
  "message": "No ARIMA params for UNKNOWN"
}
```

---

### 5. **GET** `/get_forecast`

**Параметры** (query):
- `ticker: str`

**Описание**  
- Возвращает сохранённые прогнозы (actual / predicted) по заданному тикеру из `df_forecasts`.

**Пример запроса**  
```http
GET /get_forecast?ticker=AAPL
```
**Пример ответа**  
```json
{
  "status": "ok",
  "ticker": "AAPL",
  "rows": [
    { "date": "2023-01-01", "actual": 100.5, "predicted": 101.0 },
    { "date": "2023-01-02", "actual": 101.0, "predicted": 100.8 }
  ],
  "message": "Forecast for AAPL"
}
```
Если нет данных:
```json
{
  "status": "error",
  "ticker": "UNKNOWN",
  "rows": [],
  "message": "No forecast data for ticker=UNKNOWN"
}
```

---

### 6. **GET** `/get_portfolio_data`

**Описание**  
- Возвращает данные портфеля из `overal_data.xlsx`.

**Пример ответа**  
```json
{
  "status": "ok",
  "data": [
    {
      "date": "2024-01-01",
      "portfolio": 100000.0,
      "index": 3500.0
    },
    {
      "date": "2024-01-02",
      "portfolio": 101000.0,
      "index": 3520.0
    }
  ]
}
```
Если файл отсутствует / ошибка чтения:
```json
{
  "status": "error",
  "data": []
}
```

---

### 7. **GET** `/models`

**Описание**  
- Возвращает список моделей, доступных в `model_store`.

**Пример ответа**  
```json
{
  "models": [
    {
      "model_id": "pretrained_arima",
      "description": "Pretrained ARIMA model: well-tuned and ready to use"
    }
  ]
}
```

---

### 8. **POST** `/set`

**Тело** (`ModelID`):  
```json
{
  "model_id": "pretrained_arima"
}
```

**Описание**  
- Устанавливает активную модель (`active_model_id`).

**Пример ответа**  
```json
{
  "status": "ok",
  "active_model": "pretrained_arima"
}
```
Если модель не найдена:
```json
HTTP 404
No such model ...
```

---

### 9. **POST** `/upload_dataset`

**Описание**  
- Принимает CSV-файл (через `multipart/form-data`, поле `file`).
- Проверяет колонки `ticker`, `date`, `adjusted_close`.
- Сохраняет как `user_data.csv`.

**Пример ответа**  
```json
{
  "status": "success",
  "rows": 1000,
  "tickers_count": 5,
  "message": "User CSV uploaded successfully."
}
```
При ошибке валидации (нет нужных колонок) — `HTTP 400`.  
При других ошибках — `HTTP 500`.

---

### 10. **POST** `/fit_user_data`

**Параметры** (query/body):
- `horizon` (int), по умолчанию `30`
- `max_wait` (int), по умолчанию `10`
- `p` (int), `d` (int), `q` (int) — параметры, но фактически игнорируются (модель всё равно берёт (1,1,1)).

**Описание**  
- Читает `user_data.csv`.
- Для каждого тикера параллельно (через `ThreadPoolExecutor`) строит модель ARIMA (1,1,1), вычисляет прогноз.
- Если дольше, чем `max_wait`, возвращает статус `"timeout"`.

**Пример ответа**  
```json
{
  "status": "ok",
  "model_id": "user_arima",
  "processed_tickers_count": 5,
  "forecast_rows": 150,
  "time_spent": 3.21,
  "message": "The process took too long, we used default params: p,d,q(1,1,1)"
}
```
При ошибках валидации (нет файла или колонок) — `HTTP 404/400/500`.

---

### 11. **GET** `/daily_bars/tickers`

**Описание**  
- Возвращает список тикеров из `daily_bars.csv` (дневные свечи).

**Пример ответа**  
```json
{
  "status": "ok",
  "tickers": ["AAPL", "MSFT", "SPY", "TSLA"]
}
```
Если нет файла — `HTTP 404`.

---

### 12. **GET** `/daily_bars/data`

**Параметры** (query):
- `ticker: str`

**Описание**  
- Возвращает дневные свечи  по конкретному тикеру.

**Пример ответа**  
```json
{
  "status": "ok",
  "data": [
    {
      "time": "2023-01-01",
      "open": 100.5,
      "high": 101.5,
      "low": 99.8,
      "close": 100.9,
      "volume": 50000
    },
    {
      "time": "2023-01-02",
      "open": 101.0,
      "high": 102.3,
      "low": 100.0,
      "close": 101.5,
      "volume": 60000
    }
  ],
  "message": "OK"
}
```
Если нет данных по тикеру, `data` будет пустым.

---

### 13. **GET** `/user_data/tickers`

**Описание**  
- Возвращает список тикеров, найденных в `user_data.csv`.

**Пример ответа**  
```json
{
  "status": "ok",
  "tickers": ["GOOG", "TSLA", "IBM"]
}
```
Если нет `user_data.csv`, вернётся:
```json
{
  "status": "error",
  "tickers": []
}
```
(Или `HTTP 404` — в зависимости от реализации.)

---

### 14. **GET** `/user_data/data`

**Параметры** (query):
- `ticker: str`

**Описание**  
- Возвращает свечи из пользовательских данных (`user_data.csv`)

**Пример ответа**  
```json
{
  "status": "ok",
  "ticker": "GOOG",
  "data": [
    {
      "time": "2023-01-01",
      "open": 150.0,
      "high": 150.0,
      "low": 150.0,
      "close": 150.0,
      "volume": 0.0
    },
    {
      "time": "2023-01-02",
      "open": 152.0,
      "high": 152.0,
      "low": 152.0,
      "close": 152.0,
      "volume": 0.0
    }
  ],
  "message": "User data candles"
}
```

---

### 15. **GET** `/predict`

**Параметры** (query):
- `model_id: Optional[str]` — если не указано, по умолчанию `"user_arima"`.
- `ticker: Optional[str]` — если указать, вернём прогноз только по этому тикеру. Иначе — по всем тикерам.
- `horizon: int` — сколько строк (дней) взять из прогноза (по умолчанию 10).

**Описание**  
- Берёт модель из `model_store[model_id]`.
- Выдаёт **horizon** строк прогноза (`ticker, date, predicted`).

**Пример ответа**  
```json
{
  "status": "ok",
  "model_id": "user_arima",
  "ticker": "GOOG",
  "rows": 10,
  "forecast": [
    {
      "ticker": "GOOG",
      "date": "2023-02-01",
      "predicted": 153.0
    },
    ...
  ],
  "message": "Prediction done."
}
```
Если модель не найдена — `HTTP 404`.  
Если пустой `df_forecast` — `HTTP 500`.

---

#  Описание приложения для Streamlit

Ниже вы найдёте упрощённое руководство по взаимодействию со **Streamlit**-приложением, которое обращается к бэкенду (FastAPI-серверу) для анализа временных рядов (ARIMA) и визуализации данных.

---

## 1. Страницы и их функциональность

1. **Model Management**  
   - Отображает список доступных моделей (запрос к /models).  
   - Позволяет выбрать модель и установить её активной (/set).  
   - Рассказывает об ARIMA (краткая теоретическая справка).

2. **Exploratory Data Analysis (Daily Bars)**  
   - Загружает список тикеров из /daily_bars/tickers.  
   - Для выбранного тикера получает его дневные бары (/daily_bars/data).  
   - Позволяет ресемплировать (daily, weekly, monthly, yearly).  
   - Отображает графики (свечи и объём) с помощью библиотеки streamlit_lightweight_charts.  
   - Выводит таблицу первых строк, простую статистику (describe) и box-plot (plotly).

3. **ARIMA Model Page**  
   - Загружает доступные тикеры из /list_tickers.  
   - Позволяет выбрать p и q (параметры ARIMA) и вызвать /fit_arima_params.  
   - Показывает ARIMA-параметры конкретного тикера (/get_params).  
   - Отображает прогноз (actual vs. predicted) из /get_forecast, а также считает MSE, RMSE, R2.  
   - Визуализирует фактические и предсказанные значения графиком (plotly).

4. **Portfolio Builder**  
   - Загружает данные портфеля из /get_portfolio_data.  
   - Строит кумулятивные доходности (cumprod) и сравнивает «portfolio» с «index».  
   - Демонстрирует график (plotly) и таблицу статистик (через get_stats).  

5. **User Data: Upload & EDA**  
   - Предоставляет форму для загрузки файла CSV (/upload_dataset).  
   - Если данные загружены, то показывает список тикеров в user_data.csv (/user_data/tickers).  
   - Для выбранного тикера отображает свечи,  
     а также аналогично позволяет ресемплировать, строить графики, описательную статистику и box-plot.

6. **User Data: Forecast with ARIMA**  
   - Позволяет вызвать /fit_user_data, обучив ARIMA (1,1,1 или заданные p, d, q) на user_data.csv.  
   - Отображает сообщения об успехе или таймауте.  
   - Предоставляет кнопку «Show forecast data», которая вызывает /predict?model_id=user_arima и показывает получившиеся строки прогноза.

---

## 2. Настройка

- **BACKEND_URL** (или BASE_URL) должно указывать на адрес вашего FastAPI-сервера, например http://localhost:8000.  
- Убедитесь, что сервер действительно запущен.  
- При необходимости измените порты или параметры запросов.

---



