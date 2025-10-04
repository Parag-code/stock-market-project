from tensorflow.keras import layers, models

def build_multistep(seq_length, n_features, step_ahead=5):
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(step_ahead)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
