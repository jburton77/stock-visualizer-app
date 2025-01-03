import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Streamlit App Title
st.markdown("<h1 style='text-align: center;'>Stock Price Visualizer</h1>", unsafe_allow_html=True)

# Sidebar Input for Ticker Symbol
st.sidebar.header("Stock Search")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", value="BTC-USD").upper()

time_options = {"1 Year": "1y", "5 Years": "5y", "10 Years": "10y", "Maximum": "max"}
time_periods_in_years = {"1 Year": 1, "5 Years": 5, "10 Years": 10, "Maximum": 20}  # Assuming max is 20 years

time_period = st.sidebar.selectbox("Select Time Period:", options=list(time_options.keys()))

y_axis_scale = st.sidebar.selectbox("Select Y-Axis Scale:", options=["Standard", "Logarithmic"])

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

        # Linear Regression Model
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        linear_trendline = linear_model.predict(X)

        # Logarithmic Regression Model
        X_log = np.log1p(X)  # Apply log transformation
        log_model = LinearRegression()
        log_model.fit(X_log, y)
        log_trendline = log_model.predict(X_log)

        # Polynomial Regression Model (degree 2 for quadratic)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        poly_trendline = poly_model.predict(X_poly)

        # Equation of the trendlines
        linear_slope = linear_model.coef_[0]
        linear_intercept = linear_model.intercept_
        linear_equation = f"y = {linear_slope:.2f}x + {linear_intercept:.2f}"

        log_slope = log_model.coef_[0]
        log_intercept = log_model.intercept_
        log_equation = f"y = {log_slope:.2f}ln(x) + {log_intercept:.2f}"

        poly_coeffs = poly_model.coef_
        poly_intercept = poly_model.intercept_
        poly_equation = f"y = {poly_coeffs[2]:.2f}x² + {poly_coeffs[1]:.2f}x + {poly_intercept:.2f}"

        # Future Projections
        projection_length = int(0.25 * time_periods_in_years[time_period] * 252)  # Approx. 252 trading days per year
        future_indices = np.arange(len(data), len(data) + projection_length).reshape(-1, 1)
        future_indices_df = pd.DataFrame(future_indices, columns=['Index'])  # Add feature names
        future_linear_trendline = linear_model.predict(future_indices_df)
        future_log_trendline = log_model.predict(np.log1p(future_indices_df))  # Use log transformation with feature names
        future_poly_trendline = poly_model.predict(poly.transform(future_indices_df))  # Polynomial features

        # Plot the Data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Close'], label="Close Price", color="blue")
        ax.plot(data.index, data['10_SMA'], label="10-Day SMA", color="green")
        ax.plot(data.index, data['200_SMA'], label="200-Day SMA", color="orange")
        ax.plot(data.index, linear_trendline, label="Linear Trendline", color="red", linestyle="--")
        ax.plot(data.index, log_trendline, label="Logarithmic Trendline", color="purple", linestyle="--")
        ax.plot(data.index, poly_trendline, label="Polynomial Trendline", color="brown", linestyle="--")

        # Future Projections Plot
        future_dates = pd.date_range(start=data.index[-1], periods=projection_length + 1, freq='B')[1:]
        ax.plot(future_dates, future_linear_trendline, label="Projected Linear Trendline", color="red", linestyle="dotted")
        ax.plot(future_dates, future_log_trendline, label="Projected Logarithmic Trendline", color="purple", linestyle="dotted")
        ax.plot(future_dates, future_poly_trendline, label="Projected Polynomial Trendline", color="brown", linestyle="dotted")

        # Set Y-Axis Scale
        if y_axis_scale == "Logarithmic":
            ax.set_yscale("log")

        # Chart Customizations
        ax.set_title(f"Stock Price for {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Show Trendline Equations
        plt.text(0.05, 0.90, linear_equation, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.text(0.05, 0.85, log_equation, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.text(0.05, 0.80, poly_equation, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.text(0.20, 0.95, "Equations of Regression Trendlines", transform=plt.gca().transAxes, fontsize=10,
                 ha='center', va='center', fontweight='bold')

        st.pyplot(fig)

    except Exception as e:
        st.error("Could not retrieve stock data. Please check the ticker symbol and try again.")