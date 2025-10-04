import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import preprocess
from model_classification import build_classification
import numpy as np

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


# Load preprocessed data
X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(step_ahead=1)

# Flatten y arrays so they are 1D
y_train = y_train.flatten()
y_val = y_val.flatten()
y_test = y_test.flatten()

# Create classification labels → 1 if price increased, else 0
y_train_class = np.where(np.diff(y_train, prepend=y_train[0]) > 0, 1, 0)
y_val_class = np.where(np.diff(y_val, prepend=y_val[0]) > 0, 1, 0)
y_test_class = np.where(np.diff(y_test, prepend=y_test[0]) > 0, 1, 0)

# Build and train model
model = build_classification(seq_length=60, n_features=X_train.shape[2])
model.fit(X_train, y_train_class, validation_data=(X_val, y_val_class), epochs=15, batch_size=32)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/stock_classification.h5")
print("✅ Classification model trained & saved successfully.")
