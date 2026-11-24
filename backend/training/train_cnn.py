import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

torch.set_num_threads(8)

from backend.datasets.dff_dataset import get_cnn_dataloader
from backend.models.cnn_model import FakeLensSimpleCNN

#train the model for one complete pass through the training dataset
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    total_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader, 1):
        images = batch["image"].to(device)
        targets = batch["binary_label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

        if batch_idx % 50 == 0 or batch_idx == total_batches:
            avg_loss_so_far = total_loss / (batch_idx * images.size(0))
            print(f"    Batch [{batch_idx:3d}/{total_batches}] - Running Loss: {avg_loss_so_far:.4f}", flush=True)

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_prec = precision_score(all_targets, all_preds, average="binary", zero_division=0)
    epoch_rec = recall_score(all_targets, all_preds, average="binary", zero_division=0)
    epoch_f1 = f1_score(all_targets, all_preds, average="binary", zero_division=0)
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1


#check the model performance on the selected dataset split
def evaluate_split(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["binary_label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    total_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="binary", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    return {
        "loss": total_loss,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "preds": all_preds,
        "targets": all_targets
    }

#manage the complete cnn training process, including validation
def train_cnn():
    print("=" * 65, flush=True)
    print(" FakeLens: Training Binary CNN (FakeLensSimpleCNN) on CPU", flush=True)
    print(f" CPU threads allocated: {torch.get_num_threads()}", flush=True)
    print("=" * 65, flush=True)

    root = Path(__file__).resolve().parent.parent.parent
    results_dir = root / "results"
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = results_dir / "train_manifest.csv"
    val_manifest = results_dir / "val_manifest.csv"
    test_manifest = results_dir / "test_manifest.csv"
    history_csv = results_dir / "cnn_training_history.csv"

    device = torch.device("cpu")
    batch_size = 16
    epochs = 5
    lr = 1e-4

    print(f"\n[1] Initializing DataLoaders (batch_size={batch_size})...", flush=True)
    train_loader = get_cnn_dataloader(train_manifest, root_dir=root, split="train", batch_size=batch_size, num_workers=0)
    val_loader = get_cnn_dataloader(val_manifest, root_dir=root, split="val", batch_size=batch_size, num_workers=0)
    test_loader = get_cnn_dataloader(test_manifest, root_dir=root, split="test", batch_size=batch_size, num_workers=0)

    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}", flush=True)

    torch.manual_seed(42)
    model = FakeLensSimpleCNN(num_classes=2, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")
    best_checkpoint = models_dir / "cnn_best.pt"
    latest_checkpoint = models_dir / "cnn_checkpoint_latest.pt"
    history = []
    start_epoch = 1

    if latest_checkpoint.exists():
        print(f"\n[!] Detected latest checkpoint at: {latest_checkpoint.name}", flush=True)
        try:
            ckpt = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", [])
            saved_epoch = ckpt["epoch"]
            start_epoch = saved_epoch + 1
            print(f"    Resuming from completed Epoch {saved_epoch} (Best Val Loss so far: {best_val_loss:.4f})", flush=True)
        except Exception as e:
            print(f"    Warning: Could not restore checkpoint ({e}). Starting fresh.", flush=True)

    if start_epoch > epochs:
        print(f"Model already finished all {epochs} epochs! Proceeding directly to evaluation.", flush=True)
    else:
        print(f"\n[2] Beginning Training: Epochs {start_epoch} to {epochs} (Best model selected by Val Loss)...", flush=True)
        t0 = time.time()

        for epoch in range(start_epoch, epochs + 1):
            ep_start = time.time()
            print(f"\n--- Epoch {epoch}/{epochs} ---", flush=True)
            tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_res = evaluate_split(model, val_loader, criterion, device)
            ep_time = time.time() - ep_start

            scheduler.step(val_res["loss"])

            print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] | "
                  f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}, F1: {tr_f1:.4f} | "
                  f"Val Loss: {val_res['loss']:.4f}, Acc: {val_res['accuracy']:.4f}, Prec: {val_res['precision']:.4f}, Rec: {val_res['recall']:.4f}, F1: {val_res['f1']:.4f}", flush=True)

            history.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "train_precision": tr_prec,
                "train_recall": tr_rec,
                "train_f1": tr_f1,
                "val_loss": val_res["loss"],
                "val_acc": val_res["accuracy"],
                "val_precision": val_res["precision"],
                "val_recall": val_res["recall"],
                "val_f1": val_res["f1"]
            })

            is_best = val_res["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_res["loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_res
                }, best_checkpoint)
                print(f"  [*] Validation loss improved to {best_val_loss:.4f} -> Saved best model to {best_checkpoint.name}", flush=True)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history
            }, latest_checkpoint)
            print(f"  [+] Resumable state saved to {latest_checkpoint.name}", flush=True)

            pd.DataFrame(history).to_csv(history_csv, index=False)
            print(f"  [+] Training history updated in {history_csv.name}", flush=True)

        total_training_time = time.time() - t0
        print(f"\nTraining finished in {total_training_time:.1f}s (~{total_training_time/60:.1f} mins).", flush=True)
        
    print("\n[3] Evaluating selected best checkpoint on Held-Out Test Split (1,200 images)...", flush=True)
    if not best_checkpoint.exists():
        print(f"Error: Best checkpoint {best_checkpoint} not found. Using current model state.", flush=True)
        best_epoch = epochs
    else:
        ckpt = torch.load(best_checkpoint, map_location=device, weights_only=False)
        best_epoch = ckpt["epoch"]
        print(f"Loaded best checkpoint from Epoch {best_epoch} (Validation Loss: {ckpt['best_val_loss']:.4f})", flush=True)
        model.load_state_dict(ckpt["model_state_dict"])

    test_res = evaluate_split(model, test_loader, criterion, device)

    print("\n" + "=" * 65, flush=True)
    print(" FakeLens Binary CNN Final Held-Out Test Results", flush=True)
    print("=" * 65, flush=True)
    print(f"Best Validation Epoch: {best_epoch}", flush=True)
    print(f"Test Accuracy:         {test_res['accuracy']:.4f}", flush=True)
    print(f"Test Precision:        {test_res['precision']:.4f}", flush=True)
    print(f"Test Recall:           {test_res['recall']:.4f}", flush=True)
    print(f"Test F1-Score:         {test_res['f1']:.4f}", flush=True)
    print("Confusion Matrix (Rows=True [REAL=0, FAKE=1], Cols=Pred):", flush=True)
    print(test_res["confusion_matrix"], flush=True)

    # Save CNN metrics
    cnn_test_df = pd.DataFrame([{
        "Model": "CNN (FakeLensSimpleCNN)",
        "Selected_Epoch": best_epoch,
        "Accuracy": test_res["accuracy"],
        "Precision": test_res["precision"],
        "Recall": test_res["recall"],
        "F1-Score": test_res["f1"],
        "Confusion_Matrix": test_res["confusion_matrix"].tolist()
    }])
    cnn_test_df.to_csv(results_dir / "cnn_evaluation.csv", index=False)
    print(f"Saved evaluation metrics to: {results_dir / 'cnn_evaluation.csv'}", flush=True)

    # Save test predictions for per-generator analysis
    test_manifest_df = pd.read_csv(test_manifest)
    test_manifest_df["cnn_pred"] = test_res["preds"]
    test_manifest_df.to_csv(results_dir / "cnn_test_predictions.csv", index=False)
    print(f"Saved test predictions to: {results_dir / 'cnn_test_predictions.csv'}", flush=True)

    # Per-generator breakdown
    print("\nPer-Generator Accuracy Breakdown on Test Set:", flush=True)
    src_col = "source_class" if "source_class" in test_manifest_df.columns else "source"
    for src in sorted(test_manifest_df[src_col].unique()):
        src_subset = test_manifest_df[test_manifest_df[src_col] == src]
        target_num = 0 if src == "wiki" else 1
        src_acc = (src_subset["cnn_pred"] == target_num).mean()
        print(f"  - {src:12s} ({len(src_subset):4d} images): {src_acc * 100:.2f}% accuracy", flush=True)

    print("\n[✓] All CNN training and evaluation steps successfully completed!", flush=True)


if __name__ == "__main__":
    train_cnn()
