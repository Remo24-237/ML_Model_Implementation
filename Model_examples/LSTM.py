import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf  # Library for fetching stock data
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

# Function to create sequences for LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Function to build the LSTM model
def build_lstm_model(sequence_length, n_features):
    model = Sequential()
    model.add(LSTM(units=50, activation='sigmoid', input_shape=(sequence_length, n_features)))
    model.add(Dense(units=5))
    model.add(Dense(units=3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the LSTM model
def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Function to make predictions using the trained model
def make_predictions(model, X_test, scaler):
    y_pred = model.predict(X_test)
    y_pred_actual = scaler.inverse_transform(y_pred)
    return y_pred_actual

# Function to plot the actual vs predicted stock prices
def plot_results(y_test_actual, y_pred_actual):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label='Actual')
    plt.plot(y_pred_actual, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel(f'{stock_symbol}')
    plt.ylabel('Stock Price')
    plt.show()

# Specify stock index and date range
stock_symbol = 'AAPL'
start_date = '2019-01-01'
end_date = '2024-03-16'

# Get stock data
stock_data = get_stock_data(stock_symbol, start_date, end_date)

# Preprocess data
preprocessed_data, scaler = preprocess_data(stock_data)

# Hyperparameters
sequence_length = 10
n_features = 1
epochs = 75
batch_size = 32

# Create sequences for LSTM
X, y = create_sequences(preprocessed_data['ScaledPrice'].values.reshape(-1, 1), sequence_length)

# Splitting data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build LSTM model
model = build_lstm_model(sequence_length, n_features)

# Training the model
train_model(model, X_train, y_train, epochs, batch_size)

# Making predictions
y_pred_actual = make_predictions(model, X_test, scaler)

# Inverse transform the test data to get actual stock prices
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotting actual vs predicted values
plot_results(y_test_actual, y_pred_actual)

# Make predictions on the test set
predictions = model.predict(X_test)

# Threshold the predictions for binary classification
binary_predictions = (predictions > 0.5).astype(int)

# Evaluate the model
accuracy = np.sum(binary_predictions.flatten() == y_test) / len(y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')