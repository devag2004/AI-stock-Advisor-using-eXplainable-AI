import mplfinance as mpf
import pandas as pd
import yfinance as yf

# date range
start_date = "2019-01-01"
end_date = None


reliance = yf.Ticker("RELIANCE.NS")
reliance_data = reliance.history(start=start_date, end=end_date)
print("Reliance Industries Data:")
print(reliance_data.head())

reliance_info = reliance.info
print("\nReliance Industries Fundamental Data:")
print(reliance_info)

nifty50 = yf.Ticker("^NSEI")
nifty50_data = nifty50.history(start=start_date, end=end_date)
print("\nNIFTY 50 Index Data:")
print(nifty50_data.head())
reliance_data.to_csv("reliance_data.csv")
nifty50_data.to_csv("nifty50_data.csv")
print("Missing values in Reliance Industries Data:")
print(reliance_data.isnull().sum())

print("\nMissing values in NIFTY 50 Index Data:")
print(nifty50_data.isnull().sum())

import pandas as pd
import pandas_ta as ta

reliance_data = pd.read_csv("reliance_data.csv", index_col=0, parse_dates=True)
nifty50_data = pd.read_csv("nifty50_data.csv", index_col=0, parse_dates=True)

def add_indicators(df):
    df = df.copy()

    # Trend Indicators
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["EMA_20"] = ta.ema(df["Close"], length=20)
    df["EMA_50"] = ta.ema(df["Close"], length=50)
    macd = ta.macd(df["Close"])
    df = df.join(macd)

    # Momentum Indicators
    df["RSI_14"] = ta.rsi(df["Close"], length=14)
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    df = df.join(stoch)

    # Volatility Indicators
    bbands = ta.bbands(df["Close"], length=20, std=2)
    df = df.join(bbands)
    df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Volume Indicator
    df["OBV"] = ta.obv(df["Close"], df["Volume"])

    return df

# Apply indicators
reliance_with_indicators = add_indicators(reliance_data)
nifty50_with_indicators = add_indicators(nifty50_data)

# Save to CSV
reliance_with_indicators.to_csv("reliance_with_indicators.csv")
nifty50_with_indicators.to_csv("nifty50_with_indicators.csv")

print("Technical indicators added and saved.")
print("\nReliance sample with indicators:")
print(reliance_with_indicators.tail(5))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

df = pd.read_csv("reliance_with_indicators.csv", index_col=0, parse_dates=True)

print(df.info())
print(df.describe().T.head(15))
