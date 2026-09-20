import os
import uuid

import torch
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from configs.config import CRNN_CONFIG
from models.crnn_word_model import build_crnn
from predict_word import load_image, decode_prediction


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "word_crnn_best.pth")
SYMBOLS_PATH = os.path.join(BASE_DIR, "outputs", "symbols.txt")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def load_characters():
    # rstrip("\n") preserves the literal space character in symbols.txt.
    with open(SYMBOLS_PATH, "r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file]


def load_model():
    characters = load_characters()
    num_classes = len(characters) + 1

    model = build_crnn(
        num_classes=num_classes,
        img_height=CRNN_CONFIG["img_height"],
        num_filters=tuple(CRNN_CONFIG["num_filters"]),
        rnn_hidden_size=CRNN_CONFIG["rnn_hidden_size"],
        rnn_num_layers=CRNN_CONFIG["rnn_num_layers"],
        dropout=CRNN_CONFIG["dropout"],
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model, characters


MODEL, CHARACTERS = load_model()


def recognize_image(image_path):
    image_tensor = load_image(
        path=image_path,
        img_height=CRNN_CONFIG["img_height"],
        target_width=CRNN_CONFIG["img_width"],
        threshold=200,
    )

    with torch.no_grad():
        log_probs = MODEL(image_tensor)
        prediction = log_probs.argmax(dim=2).squeeze(1)

    return decode_prediction(prediction, CHARACTERS)


@app.route("/", methods=["GET", "POST"])
def index():
    recognized_text = None
    image_name = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please choose an image file first."

        elif not allowed_file(file.filename):
            error = (
                "Unsupported file type. "
                "Upload PNG, JPG, JPEG, BMP, or WEBP."
            )

        else:
            safe_name = secure_filename(file.filename)

            if not safe_name:
                error = "The uploaded filename is not valid."

            else:
                extension = safe_name.rsplit(".", 1)[1].lower()

                # Unique file name prevents uploaded files from overwriting each other.
                image_name = f"{uuid.uuid4().hex}.{extension}"

                image_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    image_name,
                )

                file.save(image_path)

                try:
                    recognized_text = recognize_image(image_path)

                    if recognized_text == "":
                        recognized_text = (
                            "No readable text detected. "
                            "Try a clearer image with dark text on a light background."
                        )

                except Exception as exception:
                    error = f"Prediction failed: {exception}"
                    image_name = None

    return render_template(
        "index.html",
        recognized_text=recognized_text,
        image_name=image_name,
        error=error,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="Image is too large. Please upload an image smaller than 8 MB.",
        recognized_text=None,
        image_name=None,
    ), 413


if __name__ == "__main__":
    app.run(debug=True)