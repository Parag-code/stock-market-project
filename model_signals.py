from tensorflow.keras import layers, models

def build_signals(seq_length, n_features):
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
