"""
CIFAKE Binary Image Classification Project
Neural Network Architectures for CNN and MLP Baselines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAKECNN(nn.Module):
    """
    Custom Convolutional Neural Network for CIFAKE 32x32x3 Binary Image Classification.
    Architecture:
      Input (3, 32, 32)
      -> ConvBlock 1: Conv2d(3, 32, 3, pad=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2, 2) [32, 16, 16]
      -> ConvBlock 2: Conv2d(32, 64, 3, pad=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2, 2) [64, 8, 8]
      -> ConvBlock 3: Conv2d(64, 128, 3, pad=1) -> BatchNorm2d(128) -> ReLU -> MaxPool2d(2, 2) [128, 4, 4]
      -> Flatten [2048]
      -> FC1: Linear(2048, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.4)
      -> Output: Linear(128, 1) -> Raw Logit (for BCEWithLogitsLoss)
    """
    def __init__(self, dropout_rate: float = 0.4):
        super(CIFAKECNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x16x16

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x8x8

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 128x4x4

        # Classification Head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128, bias=False)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(128, 1)

    # Forward pass computing class logits
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Head
        x = self.flatten(x)
        x = self.dropout(F.relu(self.bn_fc(self.fc1(x))))
        logits = self.fc_out(x)
        return logits.squeeze(-1)


class CIFAKEMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for CIFAKE Flattened RGB Features.
    Architecture:
      Input (3072)
      -> FC1: Linear(3072, 1024) -> BatchNorm1d(1024) -> ReLU -> Dropout(0.3)
      -> FC2: Linear(1024, 512) -> BatchNorm1d(512) -> ReLU -> Dropout(0.3)
      -> FC3: Linear(512, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.2)
      -> Output: Linear(128, 1) -> Raw Logit
    """
    def __init__(self, input_dim: int = 3072):
        super(CIFAKEMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(512, 128, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(128, 1)

    # Forward pass computing class logits
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        logits = self.fc_out(x)
        return logits.squeeze(-1)
