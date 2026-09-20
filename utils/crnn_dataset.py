# utils/crnn_dataset.py

import os
import random

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch.utils.data import Dataset


WORDS = [
    "a", "an", "and", "are", "at", "be", "but", "by",
    "cat", "code", "data", "deep", "dog", "for", "from",
    "hello", "image", "in", "is", "learn", "learning",
    "machine", "model", "neural", "of", "on", "python",
    "read", "recognition", "text", "the", "this", "to",
    "what", "with", "word", "world", "you",
]


def get_available_fonts():
    """
    Returns Windows font paths that exist on the local machine.
    Arial is normally available on Windows.
    """

    candidate_paths = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\ariali.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        r"C:\Windows\Fonts\courbd.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        r"C:\Windows\Fonts\verdanab.ttf",
        r"C:\Windows\Fonts\comic.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
    ]

    available = [path for path in candidate_paths if os.path.exists(path)]

    return available


class SyntheticLineDataset(Dataset):
    """
    Creates varied synthetic line images and CTC target sequences.

    Each result is:
        image_tensor:  (1, H, W)
        target:        (max_label_len,)
        input_length:  placeholder (recomputed from model output in training)
        target_length: scalar tensor
    """

    def __init__(
        self,
        characters,
        num_samples=10000,
        img_height=64,
        img_width=256,
        max_len=20,
    ):
        self.characters = characters
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.max_len = max_len

        self.char_to_index = {
            character: index + 1
            for index, character in enumerate(characters)
        }

        self.font_paths = get_available_fonts()

        if not self.font_paths:
            raise FileNotFoundError(
                "No Windows TrueType fonts were found. "
                "Check C:\\Windows\\Fonts or add valid font paths "
                "in get_available_fonts() in utils/crnn_dataset.py."
            )

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return self.num_samples

    def generate_text(self):
        """
        Generate one to three words, constrained by max_len.
        """

        while True:
            word_count = random.randint(1, 3)
            text = " ".join(random.choices(WORDS, k=word_count))

            if len(text) <= self.max_len:
                return text

    def choose_font(self):
        font_path = random.choice(self.font_paths)
        font_size = random.randint(24, 40)

        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            return ImageFont.load_default()

    def render_text(self, text):
        """
        Draw black/near-black text on a near-white background,
        with controlled realistic variation.
        """

        background = random.randint(245, 255)

        image = Image.new(
            "L",
            (self.img_width, self.img_height),
            color=background,
        )

        draw = ImageDraw.Draw(image)
        font = self.choose_font()

        # Pillow textbbox returns the actual rendered text bounds.
        bbox = draw.textbbox((0, 0), text, font=font)

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # If a phrase is too wide, select a smaller font until it fits.
        attempts = 0
        while text_width > self.img_width - 20 and attempts < 8:
            smaller_size = max(16, font.size - 3)
            font = ImageFont.truetype(random.choice(self.font_paths), smaller_size)

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            attempts += 1

        available_x = max(4, self.img_width - text_width - 8)
        available_y = max(4, self.img_height - text_height - 8)

        x = random.randint(4, available_x)
        y = random.randint(2, available_y)

        ink = random.randint(0, 45)

        draw.text(
            (x, y),
            text,
            fill=ink,
            font=font,
        )

        # Mild image variation.
        if random.random() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(0.85, 1.25)
            )

        if random.random() < 0.30:
            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(0.0, 0.5)
                )
            )

        # Add small grayscale noise to mimic image compression/scanning.
        if random.random() < 0.35:
            pixels = image.load()

            for _ in range(random.randint(30, 120)):
                px = random.randint(0, self.img_width - 1)
                py = random.randint(0, self.img_height - 1)

                original = pixels[px, py]
                change = random.randint(-15, 15)

                pixels[px, py] = max(0, min(255, original + change))

        return image

    def encode_text(self, text):
        """
        Map characters into target IDs.
        Index 0 remains reserved for the CTC blank token.
        """

        encoded = [
            self.char_to_index[character]
            for character in text
        ]

        target_length = len(encoded)

        target = torch.zeros(self.max_len, dtype=torch.long)
        target[:target_length] = torch.tensor(
            encoded,
            dtype=torch.long,
        )

        return target, torch.tensor(target_length, dtype=torch.long)

    def __getitem__(self, index):
        text = self.generate_text()
        image = self.render_text(text)

        image_tensor = self.to_tensor(image)

        target, target_length = self.encode_text(text)

        # Placeholder: exact CTC input lengths are measured from model output.
        input_length = torch.tensor(0, dtype=torch.long)

        return image_tensor, target, input_length, target_length