import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import joblib as jb
import talib as tb
# Set page config
st.set_page_config(page_title="Stock Price Prediction App", layout="wide")

# Title
st.title("Stock Price Prediction App")

# Sidebar
st.sidebar.header("User Input")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., APPL)")
prediction_days = st.sidebar.slider("Prediction Days", 30, 90, 60)
future_days = st.sidebar.slider("Future Days to Predict", 1, 60, 30)

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
# Calculate 20-day, 50_days and for short and medium terms analysis
data['SMA20'] = tb.SMA(data['Close'], timeperiod=20)
data['SMA50']  = tb.SMA(data['Close'], timeperiod=50)
# calculate additional indicators
data['EMA12']  = tb.EMA(data['Close'], timeperiod=12)
data['EMA26'] = tb.EMA(data['Close'], timeperiod=26)
data['Bollinger_Upper'],data['Bollinger_Middle'] ,data['Bollinger_Lower'] = tb.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
data['ATR'] = tb.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

data = data.drop(columns= ['Bollinger_Middle','Low','High']) # model does not use these columns

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
    #streamlit run app.py
