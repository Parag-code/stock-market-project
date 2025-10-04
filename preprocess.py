import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length=60, step_ahead=1):
    X, y = [], []
    for i in range(len(data) - seq_length - step_ahead):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+step_ahead, 0])
    return np.array(X), np.array(y)

def preprocess(file="data/stock_with_indicators.csv", seq_length=60, step_ahead=1):
    df = pd.read_csv(file, index_col=0)
    features = ["Close","SMA_20","EMA_20","RSI","MACD","MACD_Signal","Boll_High","Boll_Low"]
    data = df[features].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = create_sequences(scaled, seq_length, step_ahead)

    train_size = int(0.7 * len(X))
    val_size = int(0.2 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
