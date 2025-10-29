import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist['loss'], label='Training Loss')
    plt.plot(epochs, hist['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # MAE
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist['mae'], label='Training MAE')
    plt.plot(epochs, hist['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Training vs Validation MAE')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(model, X_test, y_test):
    preds = model.predict(X_test).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', label='Ideal Fit')

    plt.xlabel('Actual Health Score')
    plt.ylabel('Predicted Health Score')
    plt.title('Actual vs Predicted Health Scores')
    plt.legend()
    plt.grid(True)
    plt.show()