# cifake model architectures for 32x32 image classification

import torch
import torch.nn as nn
import torch.nn.functional as F


# 3-block cnn baseline for 32x32 cifake images
class CIFAKECNN(nn.Module):
    def __init__(self, dropout_rate: float = 0.4):
        super(CIFAKECNN, self).__init__()

        # block 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # classification head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128, bias=False)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(128, 1)

    # compute logits for binary classification
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(F.relu(self.bn_fc(self.fc1(x))))
        x = self.fc_out(x)
        return x.squeeze(-1)


# 4-layer mlp baseline for flattened 3072-d pixel vectors
class CIFAKEMLP(nn.Module):
    def __init__(self, input_dim: int = 3072, dropout_rate: float = 0.3):
        super(CIFAKEMLP, self).__init__()

        # dense layers with batchnorm and dropout
        self.fc1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(256, 64, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(64, 1)

    # forward pass for mlp binary prediction
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.fc_out(x)
        return x.squeeze(-1)
