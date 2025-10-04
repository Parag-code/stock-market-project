import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import preprocess
from model_signals import build_signals
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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


X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(step_ahead=1)

signals = []
for i in range(len(y_train)-1):
    if y_train[i+1] > y_train[i] * 1.01:
        signals.append(2)  # Buy
    elif y_train[i+1] < y_train[i] * 0.99:
        signals.append(0)  # Sell
    else:
        signals.append(1)  # Hold

encoder = OneHotEncoder(sparse_output=False)
y_signals = encoder.fit_transform(np.array(signals).reshape(-1,1))
X_train = X_train[:len(y_signals)]

model = build_signals(seq_length=60, n_features=X_train.shape[2])
model.fit(X_train, y_signals, epochs=20, batch_size=32)

model.save("models/stock_signals.h5")
print("✅ Signals model trained & saved.")
