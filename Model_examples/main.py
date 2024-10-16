import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Download historical stock data from Yahoo Finance
stock_symbol = 'AAPL'  # You can change this to any stock symbol
stock_data = yf.download(stock_symbol, start='2023-01-01', end='2024-01-01')

# Feature engineering
stock_data['Month'] = stock_data.index.month
stock_data['Day'] = stock_data.index.day
stock_data['DayOfWeek'] = stock_data.index.dayofweek

# Splitting data into training and testing sets
X = stock_data[['Month', 'Day', 'DayOfWeek']]
y = stock_data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.title(f'Actual vs Predicted Stock Price for {stock_symbol}')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.xticks(rotation=45)
plt.show()
