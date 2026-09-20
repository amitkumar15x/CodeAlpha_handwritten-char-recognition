# utils/visualize_utils.py
import matplotlib.pyplot as plt


def plot_training_history(history, save_path="training_curves.png"):
    """
    Plot accuracy and loss curves from Keras History object.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()