from preprocess import preprocess
from model_regression import build_regression

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

model = build_regression(seq_length=60, n_features=X_train.shape[2])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

model.save("models/stock_regression.h5")
print("✅ Regression model trained & saved.")
