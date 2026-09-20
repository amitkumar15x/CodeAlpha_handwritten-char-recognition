# predict_word.py

import argparse

import torch
import torchvision.transforms as T
from PIL import Image

from configs.config import CRNN_CONFIG
from models.crnn_word_model import build_crnn


def load_image(
    path,
    img_height=64,
    target_width=256,
    threshold=200,
):
    """
    Load an image and create a tensor of shape:

        (1, 1, img_height, target_width)

    For external images:
    - finds dark text on a light background,
    - crops excess whitespace,
    - preserves aspect ratio,
    - places the resized text on a white 64x256 canvas.
    """

    image = Image.open(path).convert("L")

    print("Original image size:", image.size)

    # Create a mask where dark pixels represent text/ink.
    text_mask = image.point(
        lambda pixel: 255 if pixel < threshold else 0
    )

    bbox = text_mask.getbbox()

    if bbox is not None:
        left, top, right, bottom = bbox

        padding = 12

        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(image.width, right + padding)
        bottom = min(image.height, bottom + padding)

        image = image.crop((left, top, right, bottom))

        print("Text crop size:", image.size)
    else:
        print(
            "Warning: no dark text detected. "
            "Using the full image."
        )

    original_width, original_height = image.size

    # Preserve aspect ratio and fit inside a 64x256 canvas.
    scale = min(
        target_width / original_width,
        img_height / original_height,
    )

    resized_width = max(1, int(original_width * scale))
    resized_height = max(1, int(original_height * scale))

    image = image.resize(
        (resized_width, resized_height),
        Image.BILINEAR,
    )

    # White canvas exactly matching the new training dimensions.
    canvas = Image.new(
        "L",
        (target_width, img_height),
        color=255,
    )

    # Center vertically. Keep a small horizontal left margin.
    paste_x = min(4, max(0, target_width - resized_width))
    paste_y = max(0, (img_height - resized_height) // 2)

    canvas.paste(image, (paste_x, paste_y))

    tensor = T.ToTensor()(canvas)

    return tensor.unsqueeze(0)


def decode_prediction(prediction, characters):
    """
    Greedy CTC decoding.

    - Index 0 is CTC blank.
    - Repeated adjacent nonblank indices collapse to one character.
    """

    decoded = []
    previous_index = None

    for index in prediction:
        index = int(index)

        if index != 0 and index != previous_index:
            decoded.append(characters[index - 1])

        previous_index = index

    return "".join(decoded)


def main():
    parser = argparse.ArgumentParser(
        description="Recognize text in a line image with CRNN + CTC."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to outputs/word_crnn_best.pth",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to a .png, .jpg, or other supported image file.",
    )

    parser.add_argument(
        "--symbols_path",
        type=str,
        required=True,
        help="Path to outputs/symbols.txt",
    )

    args = parser.parse_args()

    # Important: rstrip("\\n") preserves a space character in symbols.txt.
    # strip() would incorrectly delete it.
    with open(args.symbols_path, "r", encoding="utf-8") as file:
        characters = [
            line.rstrip("\n")
            for line in file
        ]

    num_classes = len(characters) + 1

    print("Number of characters:", len(characters))
    print("Number of model classes:", num_classes)

    model = build_crnn(
        num_classes=num_classes,
        img_height=CRNN_CONFIG["img_height"],
        num_filters=tuple(CRNN_CONFIG["num_filters"]),
        rnn_hidden_size=CRNN_CONFIG["rnn_hidden_size"],
        rnn_num_layers=CRNN_CONFIG["rnn_num_layers"],
        dropout=CRNN_CONFIG["dropout"],
    )

    checkpoint = torch.load(
        args.model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.eval()

    image_tensor = load_image(
        path=args.image_path,
        img_height=CRNN_CONFIG["img_height"],
        target_width=CRNN_CONFIG["img_width"],
        threshold=200,
    )

    print("Final image tensor shape:", tuple(image_tensor.shape))
    print("Tensor minimum:", image_tensor.min().item())
    print("Tensor maximum:", image_tensor.max().item())
    print("Tensor mean:", image_tensor.mean().item())

    with torch.no_grad():
        log_probs = model(image_tensor)

        print("Model output shape:", tuple(log_probs.shape))

        prediction = log_probs.argmax(dim=2).squeeze(1)

        print("Raw predicted indices:", prediction.tolist())

        recognized_text = decode_prediction(
            prediction,
            characters,
        )

    print("Recognized text:", repr(recognized_text))


if __name__ == "__main__":
    main()