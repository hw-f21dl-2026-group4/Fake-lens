import torch
import torch.nn as nn
import torch.nn.functional as F


# to build a 3-block convolutional neural network for cifake
class CIFAKECNN(nn.Module):
    def __init__(self, dropout_rate: float = 0.4):
        super(CIFAKECNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x16x16

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x8x8

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 128x4x4
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128, bias=False)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = self.flatten(x)
        x = self.dropout(F.relu(self.bn_fc(self.fc1(x))))
        logits = self.fc_out(x)
        return logits.squeeze(-1)


# to build a multi-layer perceptron for flattened cifake pixels
class CIFAKEMLP(nn.Module):
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        logits = self.fc_out(x)
        return logits.squeeze(-1)
