import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Streamlit App Title
st.markdown("<h1 style='text-align: center;'>Stock Price Visualizer</h1>", unsafe_allow_html=True)

# Sidebar Input for Ticker Symbol
st.sidebar.header("Stock Search")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", value="AAPL").upper()

time_options = {"1 Year": "1y", "5 Years": "5y", "10 Years": "10y", "Maximum": "max"}
time_period = st.sidebar.selectbox("Select Time Period:", options=list(time_options.keys()))

# Function to Load Stock Data
def load_data(ticker, period):
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period=period)
    return hist

# Load and Display Data
if ticker:
    try:
        data = load_data(ticker, time_options[time_period])
        data['10_SMA'] = data['Close'].rolling(window=10).mean()
        data['200_SMA'] = data['Close'].rolling(window=200).mean()

        # Prepare Data for Linear Regression
        data['Index'] = np.arange(len(data))
        X = data[['Index']]
        y = data['Close']

        model = LinearRegression()
        model.fit(X, y)
        trendline = model.predict(X)

        # Equation of the trendline
        slope = model.coef_[0]
        intercept = model.intercept_
        equation = f"y = {slope:.2f}x + {intercept:.2f}"

        # Plot the Data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Close'], label="Close Price", color="blue")
        ax.plot(data.index, data['10_SMA'], label="10-Day SMA", color="green")
        ax.plot(data.index, data['200_SMA'], label="200-Day SMA", color="orange")
        ax.plot(data.index, trendline, label="Trendline", color="red", linestyle="--")

        # Chart Customizations
        ax.set_title(f"Stock Price for {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Show Trendline Equation
        plt.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        st.pyplot(fig)

    except Exception as e:
        st.error("Could not retrieve stock data. Please check the ticker symbol and try again.")