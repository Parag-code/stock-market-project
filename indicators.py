import yfinance as yf
import pandas as pd
import ta

def download_stock(symbol="AAPL", start="2018-01-01", end="2025-01-01", save_path="data/stock_with_indicators.csv"):
    # Download stock data
    df = yf.download(symbol, start=start, end=end)
    
    # Flatten multi-index columns if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Ensure Close is a flat 1D Series
    if "Close" not in df.columns:
        raise ValueError("❌ 'Close' column not found in downloaded data!")
    
    # Extract Close as float 1D Series
    close_series = pd.Series(df["Close"].values.flatten(), index=df.index, name="Close")

    # Add technical indicators
    df["SMA_20"] = close_series.rolling(window=20).mean()
    df["EMA_20"] = close_series.ewm(span=20, adjust=False).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()

    macd = ta.trend.MACD(close=close_series)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    boll = ta.volatility.BollingerBands(close=close_series)
    df["Boll_High"] = boll.bollinger_hband()
    df["Boll_Low"] = boll.bollinger_lband()

    # Fill missing values
    df = df.fillna(method="bfill").fillna(method="ffill")

    # Save CSV
    df.to_csv(save_path)
    print(f"✅ Data downloaded and saved successfully at {save_path}")
    print(f"📊 Final Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    try:
        download_stock()
    except Exception as e:
        print("❌ Error:", e)
