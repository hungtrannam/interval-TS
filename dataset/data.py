import yfinance as yf
import pandas as pd
import numpy as np

# Load dữ liệu ETF VNM
ticker = 'VNM'
df = yf.download(ticker, start="2022-01-01", end="2025-01-01")

# Các đặc trưng định lượng KHÔNG sử dụng Close, Open trực tiếp
df['Lagged_Return_1d'] = df['High'].pct_change(1).shift(1)
df['High_Low_Range'] = df['High'] - df['Low']
df['Range_Change'] = df['High_Low_Range'].diff()
df['Volatility_5d'] = df['High_Low_Range'].rolling(window=5).std()
df['Volatility_20d'] = df['High_Low_Range'].rolling(window=20).std()

df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
df['Volume_Change'] = df['Volume'].pct_change(1)

# Chỉ giữ lại các biến có bằng chứng khoa học rõ ràng
df_model = df[[
    'Lagged_Return_1d', 'High_Low_Range', 'Range_Change',
    'Volatility_5d', 'Volatility_20d',
    'Volume', 'Volume_Change', 'Volume_MA5', 'Volume_MA20',
    'High', 'Low'
]].dropna()

df_model.to_csv("VNM_model_data.csv", index=True)
print(df_model.tail())
