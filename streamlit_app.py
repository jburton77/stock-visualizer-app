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
time_periods_in_years = {"1 Year": 1, "5 Years": 5, "10 Years": 10, "Maximum": 20}  # Assuming max is 20 years

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
        st.subheader("Trendline Equation")
        slope = model.coef_[0]
        intercept = model.intercept_
        equation = f"y = {slope:.2f}x + {intercept:.2f}"

        # Future Projections
        projection_length = int(0.25 * time_periods_in_years[time_period] * 252)  # Approx. 252 trading days per year
        future_indices = np.arange(len(data), len(data) + projection_length).reshape(-1, 1)
        future_trendline = model.predict(future_indices)

        # Upper and Lower Bound Calculations
        data['Upper_Bound'] = data['10_SMA'] + (data['10_SMA'] * 0.05)  # 5% above 10_SMA
        data['Lower_Bound'] = data['10_SMA'] - (data['10_SMA'] * 0.05)  # 5% below 10_SMA
        last_sma = data['10_SMA'].iloc[-1]
        future_upper_bound = last_sma * (1 + 0.05)
        future_lower_bound = last_sma * (1 - 0.05)

        # Plot the Data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Close'], label="Close Price", color="blue")
        ax.plot(data.index, data['10_SMA'], label="10-Day SMA", color="green")
        ax.plot(data.index, data['200_SMA'], label="200-Day SMA", color="orange")
        ax.plot(data.index, trendline, label="Trendline", color="red", linestyle="--")
        ax.plot(data.index, data['Upper_Bound'], label="Upper Bound", color="purple", linestyle="--")
        ax.plot(data.index, data['Lower_Bound'], label="Lower Bound", color="brown", linestyle="--")

        # Future Projections Plot
        future_dates = pd.date_range(start=data.index[-1], periods=projection_length + 1, freq='B')[1:]
        ax.plot(future_dates, future_trendline, label="Projected Trendline", color="red", linestyle="dotted")
        ax.fill_between(future_dates, future_lower_bound, future_upper_bound, color="gray", alpha=0.3, label="Projected Range")

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
