from tensorflow.keras import layers, models

def build_classification(seq_length, n_features):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
