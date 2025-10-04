import matplotlib.pyplot as plt

def plot_predictions(actual, predicted, title="Stock Prediction"):
    plt.figure(figsize=(12,6))
    plt.plot(actual, label="Actual", color="blue")
    plt.plot(predicted, label="Predicted", color="red")
    plt.title(title)
    plt.legend()
    plt.show()
