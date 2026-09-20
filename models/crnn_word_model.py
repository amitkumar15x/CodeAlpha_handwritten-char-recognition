# models/crnn_word_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNModel(nn.Module):
    """
    CNN -> sequence features -> bidirectional LSTM -> CTC class scores.

    Input:
        (batch, 1, height, width)

    Output:
        (time_steps, batch, num_classes)
    """

    def __init__(
        self,
        num_classes,
        num_filters=(64, 128, 256, 512),
        rnn_hidden_size=256,
        rnn_num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        c1, c2, c3, c4 = num_filters

        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling makes CNN output height exactly 1.
        # Width remains unchanged after the earlier width reductions.
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=c4,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.cnn(x)

        # x: (B, channels, 1, reduced_width)
        x = self.height_pool(x)

        # Remove the height dimension -> (B, channels, width)
        x = x.squeeze(2)

        # LSTM expects (time_steps, batch, features).
        x = x.permute(2, 0, 1)

        x, _ = self.rnn(x)

        # CTC requires log probabilities.
        x = self.fc(x)
        x = F.log_softmax(x, dim=2)

        return x


def build_crnn(
    num_classes,
    img_height=64,
    num_filters=(64, 128, 256, 512),
    rnn_hidden_size=256,
    rnn_num_layers=2,
    dropout=0.2,
):
    # img_height remains an argument so the call is consistent with training.
    # Adaptive pooling handles the feature-map height internally.
    return CRNNModel(
        num_classes=num_classes,
        num_filters=num_filters,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        dropout=dropout,
    )