"""
CIFAKE Binary Image Classification Project
Step 4: MLP Baseline Training Pipeline
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    MODELS_DIR,
    TABLES_DIR,
    RANDOM_SEED,
)
from src.models_architectures import CIFAKEMLP
from src.data_loader import get_pytorch_dataloaders
from src.validation import set_seed


# Train baseline Multilayer Perceptron on flattened image pixels
def train_mlp_baseline(
    max_epochs: int = 20,
    batch_size: int = 128,
    lr: float = 0.001,
    patience: int = 4,
):
    print("=" * 75)
    print("STEP 4: MLP BASELINE TRAINING")
    print("=" * 75)
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} (PyTorch {torch.__version__})")

    # Load DataLoaders with flattened RGB features (3072,)
    train_loader, val_loader, _ = get_pytorch_dataloaders(
        batch_size=batch_size,
        augment_train=False,  # Standard deterministic flattened features for MLP
        model_type="mlp",
    )
    print(f"Train batches: {len(train_loader):,} ({len(train_loader.dataset):,} images)")
    print(f"Val batches:   {len(val_loader):,} ({len(val_loader.dataset):,} images)")

    model = CIFAKEMLP(input_dim=3072).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    history = []
    start_total_time = time.time()

    print(f"\nStarting MLP training (Max Epochs: {max_epochs}, Early Stopping Patience: {patience})...\n")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{max_epochs:02d} [Train]", leave=False):
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * len(labels)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start
        curr_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {curr_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "lr": curr_lr,
            "epoch_time_sec": round(epoch_time, 2),
        })

        # Save Best Validation Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                MODELS_DIR / "mlp_best.pt",
            )
            print(f"  --> Saved new best MLP checkpoint (Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}. No validation loss improvement for {patience} epochs.")
                break

    total_training_time = time.time() - start_total_time
    print(f"\nMLP Training Finished in {total_training_time:.1f}s.")
    print(f"Best Validation Epoch: {best_epoch} (Loss: {best_val_loss:.4f}, Acc: {best_val_acc*100:.2f}%)")

    # Save training history
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df_history = pd.DataFrame(history)
    df_history.to_csv(TABLES_DIR / "mlp_training_history.csv", index=False)
    print(f"[SAVED] Training history written to {TABLES_DIR / 'mlp_training_history.csv'}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "total_training_time_sec": total_training_time,
        "history": history,
    }


if __name__ == "__main__":
    train_mlp_baseline()
