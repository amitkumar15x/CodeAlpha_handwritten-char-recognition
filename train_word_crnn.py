# train_word_crnn.py

import argparse
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import CRNN_CONFIG
from models.crnn_word_model import build_crnn
from utils.crnn_dataset import SyntheticLineDataset


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """
    All dataset images already have a common size (1, 64, 256),
    so normal stacking is sufficient.
    """

    images, targets, _, target_lengths = zip(*batch)

    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    target_lengths = torch.stack(target_lengths, dim=0)

    return images, targets, target_lengths


def flatten_ctc_targets(targets, target_lengths):
    """
    Converts padded target matrix:

        (batch, max_label_len)

    into the 1-D target format accepted by CTCLoss.
    """

    flattened_targets = []

    for target, length in zip(targets, target_lengths):
        length = int(length.item())
        flattened_targets.append(target[:length])

    return torch.cat(flattened_targets, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Train a CRNN model with CTC loss for text recognition."
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_samples", type=int, default=12000)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    characters = CRNN_CONFIG["characters"]
    num_classes = len(characters) + 1

    print("Character count:", len(characters))
    print("Model class count:", num_classes)
    print(
        "Training image size:",
        f'{CRNN_CONFIG["img_width"]}x{CRNN_CONFIG["img_height"]}',
    )

    model = build_crnn(
        num_classes=num_classes,
        img_height=CRNN_CONFIG["img_height"],
        num_filters=tuple(CRNN_CONFIG["num_filters"]),
        rnn_hidden_size=CRNN_CONFIG["rnn_hidden_size"],
        rnn_num_layers=CRNN_CONFIG["rnn_num_layers"],
        dropout=CRNN_CONFIG["dropout"],
    ).to(device)

    dataset = SyntheticLineDataset(
        characters=characters,
        num_samples=args.num_samples,
        img_height=CRNN_CONFIG["img_height"],
        img_width=CRNN_CONFIG["img_width"],
        max_len=CRNN_CONFIG["max_label_len"],
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    ctc_loss = nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = 0.0
        valid_batches = 0

        for images, targets, target_lengths in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            # Shape: (T, B, C)
            log_probs = model(images)

            time_steps, batch_size, _ = log_probs.shape

            # Must be <= T. Computing from output avoids the earlier
            # "Expected input_lengths ... at most T" CTC error.
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=time_steps,
                dtype=torch.long,
                device=device,
            )

            flat_targets = flatten_ctc_targets(
                targets,
                target_lengths,
            )

            loss = ctc_loss(
                log_probs,
                flat_targets,
                input_lengths,
                target_lengths,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1

        average_loss = epoch_loss / max(valid_batches, 1)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- Loss: {average_loss:.4f}"
        )

        if average_loss < best_loss:
            best_loss = average_loss

            model_path = os.path.join(
                args.output_dir,
                "word_crnn_best.pth",
            )

            torch.save(model.state_dict(), model_path)

            print(
                f"  Saved improved model "
                f"(best loss: {best_loss:.4f})"
            )

    symbols_path = os.path.join(
        args.output_dir,
        "symbols.txt",
    )

    # Write each character as a separate line.
    # For a literal space, the saved line contains " \n".
    with open(symbols_path, "w", encoding="utf-8") as file:
        for character in characters:
            file.write(character + "\n")

    print("\nTraining completed.")
    print(
        "Best model saved to:",
        os.path.join(args.output_dir, "word_crnn_best.pth"),
    )
    print("Symbols saved to:", symbols_path)


if __name__ == "__main__":
    main()