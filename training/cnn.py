import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from backend.models.cnn_model import FakeLensSimpleCNN
from preprocessing.image_preprocessing import get_image_transforms


# to instantiate cnn architecture
def create_cnn_model(num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    return FakeLensSimpleCNN(num_classes=num_classes, dropout=dropout)


# to train cnn on the dataset
def train_cnn(dataset: str, task: str, epochs: int = 20, batch_size: int = 32, lr: float = 1e-3, dry_run: bool = False):
    num_classes = 2 if task == "binary" else (4 if dataset.lower() == "deepfakeface" else 5)
    print(f"[CNN Training] Initializing {dataset.upper()} | {task.upper()} | {num_classes} Classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_cnn_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    if dataset.lower() == "artifact":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    if dry_run:
        print("[CNN Training] Dry-run verification passed: Architecture, optimizer, and loss initialized.")
        return model

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeLens Custom CNN Training")
    parser.add_argument("--dataset", type=str, choices=["deepfakeface", "artifact"], default="deepfakeface")
    parser.add_argument("--task", type=str, choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    train_cnn(args.dataset, args.task, args.epochs, args.batch_size, args.lr, dry_run=args.dry_run)
