import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta
from ta.momentum import RSIIndicator
import warnings
from statsmodels.tsa.arima.model import ARIMA

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

def arima_model_trainer(series , steps=30):
    series = series.dropna()
    history = list(series.values)
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


def search_multiple(company_name):
    results = yf.search(company_name)
    return [item["symbol"] for item in results.get("quotes", [])[:5]]

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


# -------------------- STATE --------------------
if "period" not in st.session_state:
    st.session_state.period = None

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Trading App", layout="wide")
st.title("📊 Stock Analysis Dashboard")

# -------------------- INPUTS --------------------
col1, col2, col3 = st.columns(3)

today = date.today()

with col1:
    company = st.text_input("Company Name", "Google")

    options = search_multiple(company)
  
if options:
    ticker = st.selectbox("Select Ticker", options)

with col2:
    start_date = st.date_input("Start Date", today.replace(year=today.year - 1))

with col3:
    end_date = st.date_input("End Date", today)

# -------------------- DATA --------------------
data = fetch_data(ticker, start_date, end_date, st.session_state.period)
data = preprocess_data(data)
data = compute_indicators(data)

# -------------------- METRICS --------------------
st.markdown("### 📈 Performance")
show_metrics(data)

st.markdown("---")

# -------------------- MAIN LAYOUT --------------------
left, right = st.columns([1, 4])

# ---------- FILTERS ----------
with left:
    st.markdown("### ⏳ Timeframe")

    for label, value in {
        "5D": "5d",
        "1M": "1mo",
        "3M": "3mo",
        "6M": "6mo",
        "1Y": "1y",
        "5Y": "5y"
    }.items():
        if st.button(label, use_container_width=True):
            st.session_state.period = value
            st.rerun()

# ---------- TABS ----------
with right:
    tab1, tab2, tab3 = st.tabs(["📈 Chart", "📊 Indicators", "📋 Data"])

    # -------- Chart Tab --------
    with tab1:
        st.plotly_chart(plot_candlestick(data), use_container_width=True)
        st.plotly_chart(plot_price(data), use_container_width=True)
        st.subheader("📊 Stock 30-Day Forecast")
        st.plotly_chart(plot_next_30_days(data), use_container_width=True)

    # -------- Indicators Tab --------
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RSI")
            st.plotly_chart(plot_rsi(data), use_container_width=True)

        with col2:
            st.subheader("MACD")
            st.plotly_chart(plot_macd(data), use_container_width=True)

    # -------- Data Tab --------
    with tab3:
        st.dataframe(
            data.tail(20).sort_index(ascending=False),
            use_container_width=True
        )