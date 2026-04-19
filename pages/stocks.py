import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from yfinance import Search
import plotly.graph_objects as go
from datetime import date, timedelta
from ta.momentum import RSIIndicator
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

warnings.filterwarnings("ignore")

# -------------------- FUNCTIONS --------------------

def preprocess_data(data):
    data = data.copy()

    data.columns = data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else data.columns # to get the first value of the tuple ('close','open')
    data.index = pd.to_datetime(data.index) # convert the time-series dates into datetime
    data = data.sort_index()
    data = data.dropna() # drop nan values

    return data


def fetch_data(ticker, start, end, period):
    if period:
        return yf.download(ticker, period=period) # for periodic data incolving periods.
    else:
        return yf.download(ticker, start=start, end=end + timedelta(days=1))# add +1 for current day prices as well


def compute_indicators(data):
    data = data.copy()

    # RSI
    rsi = RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi.rsi()

    # MACD
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9).mean()
    data['Histogram'] = data['MACD'] - data['Signal']

    return data


def plot_candlestick(data):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(height=500)
    return fig


def plot_price(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close'
    ))
    fig.update_layout(height=400)
    return fig


def plot_rsi(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI"))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    return fig


def plot_macd(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name="MACD"))
    fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], name="Signal"))
    fig.add_trace(go.Bar(x=data.index, y=data['Histogram'], name="Histogram", opacity=0.4))
    return fig


def show_metrics(data):
    if len(data) >= 2:
        last = float(data["Close"].iloc[-1])
        prev = float(data["Close"].iloc[-2])

        change = last - prev
        percent = (change / prev) * 100

        st.metric("Price", f"{last:.2f}", f"{percent:.2f}%")
        st.metric("Previous Close", f"{prev:.2f}")
        st.metric("Volume", f"{int(data['Volume'].iloc[-1]):,}")
    else:
        st.warning("Not enough data")

def lstm_scaler(series):
    train_size = int(len(series) * 0.8)
    train = series[:train_size].values.reshape(-1, 1)
    test = series[train_size:].values.reshape(-1, 1)
    sc = MinMaxScaler(feature_range=(0, 1))
    train_scaled = sc.fit_transform(train)
    test_scaled = sc.transform(test)
    return train_scaled, test_scaled, sc


def create_LSTM_sequences(data, window=3):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


def train_lstm(train, test, window=3, epochs=100, batch_size=32, units=50):
    X_train, y_train = create_LSTM_sequences(train, window)
    X_test, y_test = create_LSTM_sequences(test, window)
    model = Sequential([
        LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    return model, y_pred, y_test


def forecast_future_price(model, last_sequence, sc, days=30, window=3):
    sequence = last_sequence.copy()
    predictions = []
    for _ in range(days):
        X_input = sequence.reshape(1, window, 1)
        next_val = model.predict(X_input, verbose=0)[0][0]
        predictions.append(next_val)
        sequence = np.append(sequence[1:], [[next_val]], axis=0)
    return sc.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


@st.cache_resource
def get_lstm_forecast(series_values):
    series = pd.Series(series_values)
    train_scaled, test_scaled, sc = lstm_scaler(series)
    model, _, _ = train_lstm(train_scaled, test_scaled)
    last_sequence = test_scaled[-3:].copy()
    return forecast_future_price(model, last_sequence, sc)


def plot_lstm_forecast(data, future_prices):
    series = data['Close'].dropna()
    future_index = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=30,
        freq='B'
    )[:30]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series,
        mode='lines', name='Historical',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=future_index, y=future_prices,
        mode='lines+markers', name='LSTM Forecast',
        line=dict(color='green', dash='dash'),
        marker=dict(size=4)
    ))
    fig.update_layout(title="LSTM 30-Day Forecast", height=500)
    return fig


def arima_model_trainer(series , steps=30):
    series = series.dropna()
    history = list(series.values)
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


def search_multiple(company_name):
    try:
        search = Search(company_name)
        results = search.quotes

        return [item["symbol"] for item in results[:5]]
    except:
        return []

def plot_next_30_days(data):
    data = data['Close'].dropna()
    forecast = arima_model_trainer(data,steps=30)
    fig = go.Figure()
    future_index = pd.date_range(
    start=data.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq='B' 
)[1:]

    fig = go.Figure()

    # original price
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data,
        mode='lines',
        name='Original',
        line=dict(color='blue')
    ))

    # forecast
    fig.add_trace(go.Scatter(
        x=future_index,
        y=forecast,
        mode='lines',
        name='30-Day Forecast',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title="ARIMA 30-Day Forecast",
        height=500
    )

    return fig


POPULAR_STOCKS = {
    "-- Select a stock --": None,
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Wipro (WIPRO.NS)": "WIPRO.NS",
}

# -------------------- STATE --------------------
if "period" not in st.session_state:
    st.session_state.period = None
if "forecast_fig" not in st.session_state:
    st.session_state.forecast_fig = None

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Trading App", layout="wide")
st.title("Stock Analysis Dashboard")

# -------------------- INPUTS --------------------
today = date.today()

col1, col2 = st.columns(2)
with col1:
    quick_pick = st.selectbox("Stock", list(POPULAR_STOCKS.keys()))
with col2:
    if POPULAR_STOCKS[quick_pick] is None:
        company = st.text_input("Search by Name", "Google")
    else:
        st.text_input("Search by Name", POPULAR_STOCKS[quick_pick], disabled=True)
        company = None

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("Start Date", today.replace(year=today.year - 1))
with col4:
    end_date = st.date_input("End Date", today)

col5, col6, col7 = st.columns([2, 2, 1])
with col5:
    timeframe = st.selectbox("Timeframe", ["Custom", "5D", "1M", "3M", "6M", "1Y", "5Y"])
with col6:
    model_choice = st.selectbox("Forecast Model", ["ARIMA", "LSTM"])
with col7:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Forecast", use_container_width=True, type="primary")

period = None if timeframe == "Custom" else timeframe.lower().replace("d", "d").replace("m", "mo").replace("y", "y")
TIMEFRAME_MAP = {"5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
period = TIMEFRAME_MAP.get(timeframe, None)

if POPULAR_STOCKS[quick_pick]:
    ticker = POPULAR_STOCKS[quick_pick]
else:
    options = search_multiple(company)
    if options:
        ticker = st.selectbox("Select Ticker", options)
    else:
        st.warning("No results found")
        st.stop()

# -------------------- DATA --------------------
data = fetch_data(ticker, start_date, end_date, period)
data = preprocess_data(data)
data = compute_indicators(data)

# -------------------- METRICS --------------------
st.markdown("---")
m1, m2, m3 = st.columns(3)
if len(data) >= 2:
    last  = float(data["Close"].iloc[-1])
    prev  = float(data["Close"].iloc[-2])
    change = last - prev
    pct    = (change / prev) * 100
    with m1: st.metric("Price",          f"${last:.2f}", f"{pct:+.2f}%")
    with m2: st.metric("Previous Close", f"${prev:.2f}")
    with m3: st.metric("Volume",         f"{int(data['Volume'].iloc[-1]):,}")
st.markdown("---")

# -------------------- FORECAST LOGIC --------------------
if run:
    if model_choice == "ARIMA":
        st.session_state.forecast_fig = plot_next_30_days(data)
    else:
        with st.spinner("Training LSTM... this may take a minute"):
            future_prices = get_lstm_forecast(tuple(data['Close'].dropna().values))
        st.session_state.forecast_fig = plot_lstm_forecast(data, future_prices)

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["Chart", "Indicators", "Data"])

with tab1:
    chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"], label_visibility="collapsed")
    if chart_type == "Candlestick":
        st.plotly_chart(plot_candlestick(data), use_container_width=True)
    else:
        st.plotly_chart(plot_price(data), use_container_width=True)
    if st.session_state.forecast_fig is not None:
        st.subheader(f"{model_choice} 30-Day Forecast")
        st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)
    else:
        st.info("Select a model and click **Run Forecast** to see the 30-day prediction.")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("RSI")
        st.plotly_chart(plot_rsi(data), use_container_width=True)
    with c2:
        st.subheader("MACD")
        st.plotly_chart(plot_macd(data), use_container_width=True)

with tab3:
    st.dataframe(data.tail(20).sort_index(ascending=False), use_container_width=True)