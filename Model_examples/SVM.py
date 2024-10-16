import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import yfinance as yf
import matplotlib.pyplot as plt

# Function to fetch stock data using Yahoo Finance API
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to preprocess data and scale the stock prices
def preprocess_data(df):
    scaler = MinMaxScaler()
    df['ScaledPrice'] = scaler.fit_transform(df[['Close']])
    return df, scaler

# Function to plot the actual vs predicted stock prices
def plot_results(y_test_actual,y_pred_actual):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label='Actual')
    plt.plot(y_pred_actual, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Stock Prices (SVM)')
    plt.xlabel('Index')
    plt.xlabel(f'{stock_symbol}')
    plt.show()

# Create sequences for SVM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# Specify stock index and date range
stock_symbol = 'AAPL'
start_date = '2022-01-01'
end_date = '2022-12-31'

# Get stock data
stock_data = get_stock_data(stock_symbol, start_date, end_date)

# Preprocess data
preprocessed_data, scaler = preprocess_data(stock_data)

# Hyperparameters
sequence_length = 10

# Create sequences for SVM
X, y = create_sequences(preprocessed_data['ScaledPrice'].values.reshape(-1, 1), sequence_length)

# Splitting data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build SVR model
model = SVR(kernel='linear')
model.fit(X_train.reshape(-1, sequence_length), y_train.ravel())

# Making predictions
y_pred = model.predict(X_test.reshape(-1, sequence_length))

# Inverse transform the test data to get actual stock prices
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Plotting actual vs predicted values
plot_results(y_test_actual,y_pred_actual)
