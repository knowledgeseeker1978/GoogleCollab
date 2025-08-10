# 2_data_pipeline.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def download_data(ticker="AAPL", period="5y"):
    df = yf.download(ticker, period=period)
    df = df.dropna()
    return df

def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def convert_to_text(df):
    samples = []
    for date, row in df.iterrows():
        text = f"Date: {date.date()}, Open: {row['Open']:.2f}, Close: {row['Close']:.2f}, Volume: {row['Volume']}, RSI: {row['RSI']:.2f}"
        samples.append({"text": text, "label": row['Close']})
    return pd.DataFrame(samples)

if __name__ == "__main__":
    data = download_data()
    data = add_technical_indicators(data)
    text_df = convert_to_text(data)
    train, test = train_test_split(text_df, test_size=0.2, shuffle=False)
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
