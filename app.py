import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import joblib as jb
import talib as tb

# Set page config
st.set_page_config(page_title="Stock Price & Signals Recommendation App", layout="wide")

# Title
st.title("Stock Price & Signals Recommendation App")

# Sidebar
st.sidebar.header("User Input")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)")
future_days = st.sidebar.slider("Future Days to Predict", 1, 60, 7)
prediction_days = st.sidebar.slider("Prediction Days", 30, 90, 10)

# Main content
@st.cache_data
def load_data(symbol):
    df = yf.Ticker(symbol)
    return df.history(period="max")
    
data = load_data(stock_symbol)
# Calculate additional metrics
data['Daily_Return'] = data['Close'].pct_change() * 100
data['Price_Change'] = data['Close'] - data['Open']
data['Price_Change_Pct'] = abs(((data['Close'] - data['Open']) / data['Open']) * 100)
data['RSI'] = tb.RSI(data['Close'], timeperiod=14)
# Calculate 20-day, 50_days,200-days and for short , medium and long terms analysis
data['SMA20'] = tb.SMA(data['Close'], timeperiod=20)
data['SMA50']  = tb.SMA(data['Close'], timeperiod=50)
data['SMA200']  = tb.SMA(data['Close'], timeperiod=200)

# calculate additional indicators
data['EMA12']  = tb.EMA(data['Close'], timeperiod=12)
data['EMA26'] = tb.EMA(data['Close'], timeperiod=26)
data['Bollinger_Upper'],data['Bollinger_Middle'] ,data['Bollinger_Lower'] = tb.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
data['ATR'] = tb.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
data.dropna(inplace = True)
# Display raw data
st.subheader("INDICATORS")
st.write(data.tail())

# Plot line chart
st.subheader("Stock Price Chart")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index[-100:], data['Close'].tail(100), label="Close Price")
ax.set_title(f"{stock_symbol} Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

try:
    model = jb.load('model.pkl ')
    scaler = jb.load('scaler.pkl')
    class_model = jb.load('forestforest.pkl')
    scaler_class = jb.load('scaler_class.pkl')
    st.success("Pre-trained model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()
    
st.warning("THIS MODEL CAN ONLY PERFORM WITH STOCKS THAT ARE IN THE SAME CURRENCY")
st.divider()

# Make predictions
if st.button("Make Predictions"):
    # Prepare data for prediction
    real_data = data['Close'].values
    scaled_data = scaler.transform(real_data.reshape(-1, 1))
    
    # Create sequences
    x_test = []
    for i in range(prediction_days, len(scaled_data)):
        x_test.append(scaled_data[i-prediction_days:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Make prediction
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)
    
    # Align predictions with actual dates
    prediction_dates = data.index[prediction_days:]

    # Plot predictions
    st.subheader("Price Predictions")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, real_data, label="Actual Prices")
    ax.plot(prediction_dates, prediction, label="Predicted Prices", alpha=0.7)
    ax.set_title(f"{stock_symbol} Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


    # Future predictions
    last_sequence = scaled_data[-prediction_days:]
    future_predictions = []

    for _ in range(future_days):
        next_pred = model.predict(last_sequence.reshape(1, prediction_days, 1))
        future_predictions.append(next_pred[0])
        last_sequence = np.append(last_sequence[1:], next_pred)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    st.subheader(f"Future {future_days} Days Prediction")
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})
    st.write(future_df)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(future_dates, future_predictions, label="Future Predictions")
    ax.set_title(f"{stock_symbol} Future Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Price")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Stock Signals
# prepare data
data['MACD'], data['MACD_signal'], data['MACD_hist'] = tb.MACD(data['Close'],fastperiod=12, slowperiod=26, signalperiod=9) 
def label_stocks(stock_df):
    conditions = [
        # Buy if RSI < 30 and SMA20 > SMA50
        (stock_df['RSI'] < 30) & (stock_df['SMA20'] > stock_df['SMA50']),
        # Sell if RSI > 70 and SMA20 < SMA50
        (stock_df['RSI'] > 70) & (stock_df['SMA20'] < stock_df['SMA50']),
        # Buy if MACD crosses above the signal line (Bullish crossover)
        (stock_df['MACD'] > stock_df['MACD_signal']) & (stock_df['MACD'].shift(1) <= stock_df['MACD_signal'].shift(1)),
        # Sell if MACD crosses below the signal line (Bearish crossover)
        (stock_df['MACD'] < stock_df['MACD_signal']) & (stock_df['MACD'].shift(1) >= stock_df['MACD_signal'].shift(1)),
        # Buy if price is below the lower Bollinger Band (Oversold condition)
        (stock_df['Close'] < stock_df['Bollinger_Lower']),
        # Sell if price is above the upper Bollinger Band (Overbought condition)
        (stock_df['Close'] > stock_df['Bollinger_Upper'])
    ]
    
    # Define the corresponding choices
    choices = ['Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell']
    
    # Define the default value for 'Hold' if no condition is met
    default = 'Hold'
    
    # Apply the conditions and choices to create the 'Signal' column
    stock_df['Signal'] = np.select(conditions, choices, default=default)
    
    # If any of the key columns (RSI, SMA20, SMA50, SMA200, MACD, Upper_Band, Lower_Band) have NaN values, set 'Signal' to NaN
    stock_df.loc[stock_df[['RSI', 'SMA20', 'SMA50', 'SMA200', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']].isna().any(axis=1), 'Signal'] = np.nan
    
    return stock_df

data = label_stocks(data)
data['Signal'] = data['Signal'].map({'Hold': 0, 'Buy': 1, 'Sell': -1})
class_df = data[['Close','SMA20', 'SMA50','RSI', 'MACD','MACD_signal' ,'Bollinger_Lower', 'Signal']]
features_to_remove = [
    'SMA20',  # Too many moving averages create redundancy, not relevent in choosing signals
    'SMA50',
    'SMA200',
    'Bollinger_Middle',  # remove Bollinger_Middle (the same as SMA20)
    'EMA12'  # the same as EMA26 in most cases
]
df_cleaned = data.drop(columns=features_to_remove)
df_cleaned.dropna(inplace=True)
class_model = jb.load('forestforest.pkl')
scaler_class = jb.load('scaler_class.pkl')
# Make prediction
if st.button("Make Stock Signal Suggestion"):
    class_df.replace({0:"Hold", 1:"Buy", -1:"Sell"}, inplace = True)
    st.write(class_df.tail())
