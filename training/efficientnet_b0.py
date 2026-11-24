import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# to build efficientnet-b0 with classification head
def create_efficientnet_b0(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model


# to train efficientnet-b0 transfer model
def train_efficientnet_b0(dataset: str, task: str, epochs: int = 20, batch_size: int = 32, lr: float = 1e-4, dry_run: bool = False):
    num_classes = 2 if task == "binary" else (4 if dataset.lower() == "deepfakeface" else 5)
    print(f"[EfficientNet-B0 Training] Initializing {dataset.upper()} | {task.upper()} | {num_classes} Classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_efficientnet_b0(num_classes=num_classes, pretrained=False).to(device)

    if dry_run:
        print("[EfficientNet-B0 Training] Dry-run verification successful: Head replaced and model initialized.")
        return model

    print(f"[EfficientNet-B0 Training] Execution configured for {epochs} epochs on device: {device}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeLens EfficientNet-B0 Training")
    parser.add_argument("--dataset", type=str, choices=["deepfakeface", "artifact"], default="deepfakeface")
    parser.add_argument("--task", type=str, choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry_run", action="store_true", help="Validate setup without training")
    args = parser.parse_args()

    train_efficientnet_b0(args.dataset, args.task, args.epochs, args.batch_size, args.lr, dry_run=args.dry_run)
