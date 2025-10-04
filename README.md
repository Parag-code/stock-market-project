# 🧠 Stock Price Prediction & Trading Signal AI

### 📊 Industry: Finance | 🧮 Domain: Deep Learning | ⚙️ Framework: TensorFlow / Keras

---

## 🚀 Project Overview

This project is an advanced **Deep Learning–based Stock Forecasting System** that predicts future stock prices and provides **AI-driven trading recommendations**.

The model leverages **historical market data**, **technical indicators**, and **LSTM neural networks** to generate accurate predictions and actionable insights:

- 📈 Predicts **future stock prices** for the next 5 days  
- 🔮 Classifies **market trend** (UP or DOWN)  
- 💹 Generates **trading signals**: BUY ✅, HOLD ⏸️, or SELL ❌  

It’s designed for analysts, investors, and developers who want a smart AI tool for financial forecasting and automated trading decisions.

---

## 🧩 Key Features

| Category | Description |
|-----------|-------------|
| 🧮 **Multi-step Forecasting** | Predicts the next 5 days of stock prices instead of just 1 |
| 🔍 **Trend Prediction** | Classifies if the stock will go 📈 UP or 📉 DOWN |
| 💡 **Trading Signal AI** | Suggests Buy / Hold / Sell decisions based on model confidence |
| 🧠 **Multi-Model Framework** | Separate models for regression, classification, and signal generation |
| 📊 **Technical Indicators** | Uses RSI, SMA, EMA, MACD, and Bollinger Bands |
| 🎯 **Evaluation Metrics** | RMSE for regression, Accuracy for classification |
| 💾 **Pretrained Models** | Models saved as `.h5` for reuse |
| 🧹 **Clean Output** | Suppresses TensorFlow and warning logs for a neat CLI display |
| 📈 **Graph Visualization** | Plots last 30 days + next 5-day predictions + Buy/Sell markers |
| 🧩 **Modular Design** | Separate scripts for data, preprocessing, training, and prediction |

---

## 🧠 Tech Stack

| Layer | Tools / Libraries |
|-------|-------------------|
| 💻 Language | Python 3.10+ |
| 🧠 Framework | TensorFlow, Keras |
| 📊 Data Handling | Pandas, NumPy, scikit-learn |
| 💹 Data Source | Yahoo Finance API (via `yfinance`) |
| ⚙️ Indicators | `ta` (Technical Analysis Library) |
| 📈 Visualization | Matplotlib |
| 🧪 Evaluation | RMSE, Accuracy, Confusion Matrix |
| 🧭 IDE / Notebook | VS Code, Jupyter |

---

---

## 🧮 Dataset & Features

### **Data Source**
- Yahoo Finance API (`yfinance`)
- Example tickers: `AAPL`, `TSLA`, `RELIANCE.NS`, `GOOG`, `MSFT`

### **Features Used**
| Feature | Description |
|----------|-------------|
| `Open`, `High`, `Low`, `Close`, `Volume` | Core market data |
| `SMA_20`, `EMA_20` | Short-term moving averages |
| `RSI` | Relative Strength Index for momentum |
| `MACD`, `MACD_Signal` | Trend strength indicators |
| `Boll_High`, `Boll_Low` | Bollinger Bands for volatility |

---

## 🧠 User Input

```bash
📊 Enter Stock Symbol (e.g. AAPL, TSLA, RELIANCE.NS): AAPL
📅 Enter Target Date (YYYY-MM-DD): 2025-10-15

```

## 📤 AI Output

```bash
🔮 Prediction Results
Stock: AAPL
Target Date: 2025-10-15
📌 Predicted Price on 2025-10-15: 231.25
📌 Predicted Next 5-Day Prices: [221.38 217.43 220.84 217.76 211.44]
📌 Tomorrow’s Trend: 📉 DOWN
📌 Suggested Trading Signal: HOLD ⏸️

```
