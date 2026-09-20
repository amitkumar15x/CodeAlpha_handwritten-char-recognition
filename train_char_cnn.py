# train_char_cnn.py
import os
import argparse
import tensorflow as tf
from tensorflow import keras

from configs.config import CHAR_CNN_CONFIG, DEFAULT_TRAIN_CONFIG
from models.cnn_char_model import build_char_cnn
from utils.data_utils import load_mnist_data, load_emnist_byclass_data, load_emnist_bymerge_data
from utils.visualize_utils import plot_training_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "emnist_byclass", "emnist_bymerge"],
        help="Dataset to use",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    if args.dataset == "mnist":
        x_train, y_train, x_test, y_test = load_mnist_data()
    elif args.dataset == "emnist_byclass":
        x_train, y_train, x_test, y_test = load_emnist_byclass_data()
    elif args.dataset == "emnist_bymerge":
        x_train, y_train, x_test, y_test = load_emnist_bymerge_data()
    else:
        raise ValueError("Unsupported dataset")

    cfg = CHAR_CNN_CONFIG[args.dataset]
    num_classes = cfg["num_classes"]
    input_shape = cfg["input_shape"]

    # Build model
    model = build_char_cnn(input_shape=input_shape, num_classes=num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(
        args.output_dir, f"char_cnn_{args.dataset}.h5"
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    # Train
    history = model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=DEFAULT_TRAIN_CONFIG["val_split"],
        callbacks=[checkpoint, early_stop],
        verbose=2,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy on {args.dataset}: {test_acc:.4f}")

    # Save training curves
    plot_path = os.path.join(
        args.output_dir, f"training_curves_{args.dataset}.png"
    )
    plot_training_history(history, save_path=plot_path)
    print(f"Training curves saved to {plot_path}")
    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    main()