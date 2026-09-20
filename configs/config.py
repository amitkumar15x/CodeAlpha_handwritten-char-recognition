# configs/config.py

CHAR_CNN_CONFIG = {
    "mnist": {
        "num_classes": 10,
        "input_shape": (28, 28, 1),
        "class_names": [str(i) for i in range(10)],
    },
    "emnist_byclass": {
        "num_classes": 36,  # 0-9 + A-Z
        "input_shape": (28, 28, 1),
        "class_names": [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)],
    },
    "emnist_bymerge": {
        "num_classes": 47,  # 0-9 + A-Z (with some merged classes), labels 0-46
        "input_shape": (28, 28, 1),
        # Simple naming: 0-9 -> '0'-'9', 10-46 -> 'C0'-'C36'
        "class_names": [str(i) for i in range(10)] + [f"C{i}" for i in range(37)],
    },
}

DEFAULT_TRAIN_CONFIG = {
    "batch_size": 128,
    "epochs": 15,
    "learning_rate": 1e-3,
    "val_split": 0.1,
}

CRNN_CONFIG = {
    "img_height": 64,
    "img_width": 256,
    "max_label_len": 20,
    "characters": "abcdefghijklmnopqrstuvwxyz ",
    "num_filters": [64, 128, 256, 512],
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2,
    "dropout": 0.5,
}