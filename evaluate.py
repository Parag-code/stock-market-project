import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess
from visualize import plot_predictions
from sklearn.metrics import mean_squared_error

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


# ---------- Regression (Next-Day Prediction) ----------
X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(step_ahead=1)

reg_model = load_model("models/stock_regression.h5", compile=False)  # ✅ Safe loading
reg_preds = reg_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, reg_preds))
print(f"📈 Regression RMSE: {rmse:.6f}")

plot_predictions(y_test, reg_preds, "Next-Day Price Prediction")

# ---------- Multi-Step (Next 5 Days) ----------
X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(step_ahead=5)

multi_model = load_model("models/stock_multistep.h5", compile=False)  # ✅ Fixed here too
multi_preds = multi_model.predict(X_test)

print("\n🔮 Multi-step Example Prediction:")
print("Actual:", np.round(y_test[-1], 4))
print("Predicted:", np.round(multi_preds[-1], 4))
