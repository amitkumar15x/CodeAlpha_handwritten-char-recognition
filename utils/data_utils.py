# utils/crnn_dataset.py
import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


class SyntheticLineDataset(Dataset):
    """
    Synthetic handwritten-like line dataset:
    - Renders random words/sentences using fonts + slight noise.
    - Returns (image_tensor, target_ids, input_length, target_length).
    """

    def __init__(self, characters, num_samples=5000, img_height=64, max_len=50):
        super().__init__()
        self.characters = characters
        self.char2id = {ch: i + 1 for i, ch in enumerate(characters)}  # 0 = blank
        self.id2char = {i: ch for i, ch in self.char2id.items()}
        self.num_classes = len(characters) + 1

        self.num_samples = num_samples
        self.img_height = img_height
        self.max_len = max_len

        # Simple word list
        self.words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
            "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
            "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
            "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could",
            "them", "see", "other", "than", "then", "now", "look", "only", "come",
            "its", "over", "think", "also", "back", "after", "use", "two", "how",
            "our", "work", "first", "well", "way", "even", "new", "want", "because"
        ]

    def __len__(self):
        return self.num_samples

    def _render_text(self, text):
        # Approximate width based on length
        avg_char_width = 20
        width = max(100, len(text) * avg_char_width + 40)
        img = Image.new("L", (width, self.img_height), color=255)
        draw = ImageDraw.Draw(img)

        # Try to use a default font; fall back to default if not found
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except IOError:
            font = ImageFont.load_default()

        # Draw text with slight random offset
        x_offset = random.randint(5, 15)
        y_offset = random.randint(5, 10)
        draw.text((x_offset, y_offset), text, fill=0, font=font)

        # Convert PIL image to numpy array
        img_arr = np.array(img).astype("float32") / 255.0
        img_tensor = torch.from_numpy(img_arr)  # (H, W)
        img_tensor = img_tensor.unsqueeze(0)    # (1, H, W)
        return img_tensor

    def __getitem__(self, idx):
        # Generate random text
        num_words = random.randint(1, 5)
        text = " ".join(random.choices(self.words, k=num_words))
        text = text[:self.max_len]  # truncate if needed

        img = self._render_text(text)

        # Encode text to ids
        target = [self.char2id[ch] for ch in text if ch in self.char2id]
        target = torch.tensor(target, dtype=torch.long)

        input_length = torch.tensor([img.shape[-1] // 4], dtype=torch.long)  # rough
        target_length = torch.tensor([len(target)], dtype=torch.long)

        return img, target, input_length, target_length