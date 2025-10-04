import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import preprocess
from indicators import download_stock
import datetime

# ---------- CLEAN CONSOLE SETTINGS ----------
import os, warnings, tensorflow as tf

# Hide TensorFlow info & retracing messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Hide all Python warning spam (font, deprecation, etc.)
warnings.filterwarnings("ignore", message="Glyph .* missing from current font")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# --------------------------------------------


# ---------- Helper ----------
def format_date(date_str):
    """Validate date input"""
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        print("❌ Date format wrong! Please enter YYYY-MM-DD")
        exit()

def plot_results(df, next_5days_real, trade_signal, symbol, target_date):
    """Plot actual vs predicted + trading signal markers"""
    plt.figure(figsize=(12, 6))

    # Last 30 days actual prices
    recent_prices = df["Close"].values[-30:]
    plt.plot(range(len(recent_prices)), recent_prices, label="Actual (Last 30 Days)", color="blue")

    # Next 5 days predictions
    future_range = range(len(recent_prices), len(recent_prices) + 5)
    plt.plot(future_range, next_5days_real, label="Predicted (Next 5 Days)", color="red", marker="o")

    # Mark signals
    if trade_signal.startswith("BUY"):
        plt.scatter(future_range[-1], next_5days_real[-1], color="green", s=120, label="Buy Signal ✅")
    elif trade_signal.startswith("SELL"):
        plt.scatter(future_range[-1], next_5days_real[-1], color="red", s=120, label="Sell Signal ❌")
    else:
        plt.scatter(future_range[-1], next_5days_real[-1], color="gray", s=120, label="Hold Signal ⏸️")

    plt.title(f"Stock Prediction for {symbol} — {target_date}")
    plt.xlabel("Days")
    plt.ylabel("Price (USD or INR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- Main ----------
if __name__ == "__main__":
    symbol = input("📊 Enter Stock Symbol (e.g. AAPL, TSLA, RELIANCE.NS): ").upper()
    target_date = input("📅 Enter Target Date (YYYY-MM-DD): ")
    target_date = format_date(target_date)

    today = datetime.date.today()

    # Download latest stock data
    df = download_stock(symbol=symbol, start="2018-01-01", end=str(today), save_path="data/stock_with_indicators.csv")

    # ---------- 1️⃣ Next-day price ----------
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(step_ahead=1)
    reg_model = load_model("models/stock_regression.h5", compile=False)  # ✅ Fix mse bug
    reg_pred = reg_model.predict(X_test)
    day_price_scaled = reg_pred[-1][0]
   # Create dummy array with same feature dimension as training
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, 0] = day_price_scaled   # assuming first feature = Close price
    day_price_real = scaler.inverse_transform(dummy)[0][0]
# ✅ convert to actual price

    # ---------- 2️⃣ Next 5-day forecast ----------
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(step_ahead=5)
    multi_model = load_model("models/stock_multistep.h5", compile=False)  # ✅ Fix mse bug
    multi_pred = multi_model.predict(X_test)
    next_5days_scaled = multi_pred[-1]
    dummy_multi = np.zeros((len(next_5days_scaled), scaler.n_features_in_))
    dummy_multi[:, 0] = next_5days_scaled  # first feature = Close
    next_5days_real = scaler.inverse_transform(dummy_multi)[:, 0]
  # ✅ actual prices

    # ---------- 3️⃣ Trend (Up/Down) ----------
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(step_ahead=1)
    cls_model = load_model("models/stock_classification.h5", compile=False)
    cls_pred = (cls_model.predict(X_test) > 0.5).astype(int)
    trend = "📈 UP" if cls_pred[-1] == 1 else "📉 DOWN"

    # ---------- 4️⃣ Trading Signals ----------
    sig_model = load_model("models/stock_signals.h5", compile=False)
    sig_pred = sig_model.predict(X_test)
    sig_class = np.argmax(sig_pred[-1])
    signal_map = {0: "SELL ❌", 1: "HOLD ⏸️", 2: "BUY ✅"}
    trade_signal = signal_map[sig_class]

    # ---------- 💬 Show Results ----------
    print("\n🔮 Prediction Results")
    print(f"Stock: {symbol}")
    print(f"Target Date: {target_date}")
    print(f"📌 Predicted Price on {target_date}: {day_price_real:.2f}")
    print(f"📌 Predicted Next 5-Day Prices: {np.round(next_5days_real, 2)}")
    print(f"📌 Tomorrow’s Trend: {trend}")
    print(f"📌 Suggested Trading Signal: {trade_signal}")

    # ---------- 📈 Show Graph ----------
    plot_results(df, next_5days_real, trade_signal, symbol, target_date)
