# test_training_sample.py

import os

import torch
from PIL import Image

from configs.config import CRNN_CONFIG
from models.crnn_word_model import build_crnn
from utils.crnn_dataset import SyntheticLineDataset


def decode_prediction(prediction, characters):
    decoded = []
    previous_index = None

    for index in prediction:
        index = int(index)

        if index != 0 and index != previous_index:
            decoded.append(characters[index - 1])

        previous_index = index

    return "".join(decoded)


def target_to_text(target, target_length, characters):
    result = []

    for index in target[:int(target_length)]:
        index = int(index)

        if index != 0:
            result.append(characters[index - 1])

    return "".join(result)


def main():
    characters = CRNN_CONFIG["characters"]
    num_classes = len(characters) + 1

    model = build_crnn(
        num_classes=num_classes,
        img_height=CRNN_CONFIG["img_height"],
        num_filters=tuple(CRNN_CONFIG["num_filters"]),
        rnn_hidden_size=CRNN_CONFIG["rnn_hidden_size"],
        rnn_num_layers=CRNN_CONFIG["rnn_num_layers"],
        dropout=CRNN_CONFIG["dropout"],
    )

    model.load_state_dict(
        torch.load(
            "outputs/word_crnn_best.pth",
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    dataset = SyntheticLineDataset(
        characters=characters,
        num_samples=1,
        img_height=CRNN_CONFIG["img_height"],
        img_width=CRNN_CONFIG["img_width"],
        max_len=CRNN_CONFIG["max_label_len"],
    )

    image_tensor, target, _, target_length = dataset[0]

    expected_text = target_to_text(
        target,
        target_length,
        characters,
    )

    with torch.no_grad():
        log_probs = model(image_tensor.unsqueeze(0))
        prediction = log_probs.argmax(dim=2).squeeze(1)

        predicted_text = decode_prediction(
            prediction,
            characters,
        )

    print("Expected text: ", repr(expected_text))
    print("Predicted text:", repr(predicted_text))

    os.makedirs("outputs", exist_ok=True)

    image = image_tensor.squeeze(0).clamp(0, 1)
    image = (image * 255).byte()

    output_path = "outputs/generated_test_sample.png"

    Image.fromarray(image.numpy()).save(output_path)

    print("Saved image:", output_path)


if __name__ == "__main__":
    main()