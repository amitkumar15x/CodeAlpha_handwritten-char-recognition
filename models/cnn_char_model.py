# models/cnn_char_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_char_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    CNN architecture for handwritten character recognition.
    Input: (H, W, C) grayscale images.
    Output: softmax over num_classes.
    """
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Classifier head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model