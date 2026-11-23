"""
CIFAKE Binary Image Classification Project
Phase 3: CNN Controlled Optimization Pipeline
Compares:
1. CNN_3Block_AdamW_Cosine: 3-Block CNN with AdamW (lr=0.0005, wd=1e-4), CosineAnnealingLR, dropout=0.3, mild augmentation
2. CNN_Residual_AdamW_Cosine: CIFAKEResNet with AdamW (lr=0.0005, wd=1e-4), CosineAnnealingLR, dropout=0.3, mild augmentation
3. CNN_4Block_AdamW_Cosine: CIFAKECNN4Block with AdamW (lr=0.0005, wd=1e-4), CosineAnnealingLR, dropout=0.3, mild augmentation
4. CNN_3Block_NoAug_LR0005: 3-Block CNN with No Data Augmentation, AdamW, lr=0.0005
Logs all trials into results/tables/phase3_experiments_log.csv
Trains the top architecture to convergence and saves models/cnn_optimized_best.pt
"""

import sys
import time
import os
import psutil
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs.config import MODELS_DIR, TABLES_DIR, RANDOM_SEED, BATCH_SIZE
from src.models_architectures import CIFAKECNN
from src.models_architectures_phase3 import CIFAKEResNet, CIFAKECNN4Block
from src.data_loader import get_pytorch_dataloaders
from src.experiment_tracker import log_experiment
from src.validation import set_seed


# Train single CNN candidate model and track validation loss
def train_single_cnn_trial(
    trial_cfg: Dict[str, Any],
    device,
    max_epochs: int = 5,
    patience: int = 3,
) -> Dict[str, Any]:
    exp_id = trial_cfg["exp_id"]
    print("\n" + "=" * 75)
    print(f"RUNNING CNN TRIAL: {exp_id} ({trial_cfg['desc']})")
    print("=" * 75)
    set_seed(RANDOM_SEED)

    process = psutil.Process(os.getpid())
    mem_baseline = process.memory_info().rss / (1024 * 1024)

    # Load dataloaders
    augment = trial_cfg.get("augment", True)
    train_loader, val_loader, _ = get_pytorch_dataloaders(
        batch_size=BATCH_SIZE,
        augment_train=augment,
        model_type="cnn",
    )

    # Instantiate model
    arch_type = trial_cfg["arch"]
    dropout = trial_cfg.get("dropout", 0.3)
    if arch_type == "3block":
        model = CIFAKECNN(dropout_rate=dropout).to(device)
    elif arch_type == "residual":
        model = CIFAKEResNet(dropout_rate=dropout).to(device)
    elif arch_type == "4block":
        model = CIFAKECNN4Block(dropout_rate=dropout).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch_type}")

    criterion = nn.BCEWithLogitsLoss()
    lr = trial_cfg.get("lr", 0.0005)
    weight_decay = trial_cfg.get("weight_decay", 1e-4)

    if trial_cfg.get("optimizer", "adamw") == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched_type = trial_cfg.get("scheduler", "cosine")
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_metrics = {}
    best_state_dict = None
    patience_counter = 0

    history = []
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
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

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_preds = []
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * len(labels)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend((probs >= 0.5).astype(int))
                val_targets.extend(labels.cpu().numpy().astype(int))

        val_loss = val_loss_sum / len(val_targets)
        val_acc = accuracy_score(val_targets, val_preds)
        val_prec = precision_score(val_targets, val_preds, pos_label=1)
        val_rec = recall_score(val_targets, val_preds, pos_label=1)
        val_f1 = f1_score(val_targets, val_preds, pos_label=1)
        val_auc = roc_auc_score(val_targets, val_probs)
        ep_time = time.time() - ep_start

        if sched_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        print(
            f"  Epoch {epoch:02d}/{max_epochs:02d} | Train Acc: {train_acc*100:.2f}% (Loss: {train_loss:.4f}) | "
            f"Val Acc: {val_acc*100:.2f}% (Loss: {val_loss:.4f}) | Val F1: {val_f1:.4f} | AUC: {val_auc:.4f} | Time: {ep_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_auc": val_auc,
            "epoch_time": ep_time,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = {
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
                "val_auc": val_auc,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [EARLY STOPPING] Epoch {epoch}: No validation loss improvement for {patience} epochs.")
                break

    total_time = time.time() - t_start
    infer_time = ep_time * (len(val_loader) / (len(train_loader) + len(val_loader)))
    mem_now = process.memory_info().rss / (1024 * 1024)
    peak_mem = round(max(0, mem_now - mem_baseline), 1)

    print(f"Trial {exp_id} Complete -> Best Epoch: {best_epoch} | Best Val Acc: {best_val_acc*100:.2f}% | Val F1: {best_metrics['val_f1']:.4f} | Total Time: {total_time:.1f}s")

    # Log trial
    log_experiment({
        "Model": "CNN",
        "Experiment_ID": exp_id,
        "Architecture": f"CNN_{arch_type.upper()}",
        "Hyperparameters": f"opt={trial_cfg['optimizer']}, lr={lr}, wd={weight_decay}, drop={dropout}, sched={sched_type}, aug={augment}",
        "Feature_Representation": "RGB-3x32x32",
        "Train_Samples": 79707,
        "Val_Samples": 19619,
        "Train_Accuracy": best_metrics["train_accuracy"],
        "Val_Accuracy": best_metrics["val_accuracy"],
        "Val_Precision": best_metrics["val_precision"],
        "Val_Recall": best_metrics["val_recall"],
        "Val_F1": best_metrics["val_f1"],
        "Val_ROC_AUC": best_metrics["val_auc"],
        "Train_Time_Sec": total_time,
        "Infer_Time_Sec": infer_time,
        "Peak_Memory_MB": peak_mem,
        "Checkpoint_Path": f"models/cnn_optimized_{exp_id}.pt",
    })

    return {
        "exp_id": exp_id,
        "arch": arch_type,
        "best_epoch": best_epoch,
        "best_state_dict": best_state_dict,
        "best_metrics": best_metrics,
        "history": history,
        "total_time": total_time,
        "trial_cfg": trial_cfg,
    }


# Run grid evaluation across CNN architectural hyperparameters
def run_cnn_optimization():
    print("=" * 80)
    print("PHASE 3: CNN CONTROLLED OPTIMIZATION EXPERIMENTS")
    print("=" * 80)
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trials = [
        # Trial 1: Baseline 3-Block architecture + AdamW + CosineAnnealingLR + lr=0.0005
        {
            "exp_id": "CNN_3Block_AdamW_Cosine",
            "arch": "3block",
            "optimizer": "adamw",
            "lr": 0.0005,
            "weight_decay": 1e-4,
            "dropout": 0.3,
            "scheduler": "cosine",
            "augment": True,
            "desc": "3-Block CNN with AdamW + CosineAnnealingLR (lr=0.0005)",
        },
        # Trial 2: Residual CNN (CIFAKEResNet) + AdamW + CosineAnnealingLR
        {
            "exp_id": "CNN_Residual_AdamW_Cosine",
            "arch": "residual",
            "optimizer": "adamw",
            "lr": 0.0005,
            "weight_decay": 1e-4,
            "dropout": 0.3,
            "scheduler": "cosine",
            "augment": True,
            "desc": "CIFAKEResNet Residual Blocks with AdamW + Cosine",
        },
        # Trial 3: 4-Block CNN (CIFAKECNN4Block)
        {
            "exp_id": "CNN_4Block_AdamW_Cosine",
            "arch": "4block",
            "optimizer": "adamw",
            "lr": 0.0005,
            "weight_decay": 1e-4,
            "dropout": 0.3,
            "scheduler": "cosine",
            "augment": True,
            "desc": "4-Block CNN (32->64->128->256) with AdamW + Cosine",
        },
        # Trial 4: 3-Block CNN No Augmentation comparison
        {
            "exp_id": "CNN_3Block_NoAug_LR0005",
            "arch": "3block",
            "optimizer": "adamw",
            "lr": 0.0005,
            "weight_decay": 1e-4,
            "dropout": 0.3,
            "scheduler": "cosine",
            "augment": False,
            "desc": "3-Block CNN with Deterministic/No Augmentation",
        },
    ]

    best_val_acc = -1.0
    best_trial_result = None
    all_trial_summaries = []

    print(f"Executing {len(trials)} Controlled CNN Trials (4 epochs each)...\n")

    for trial_cfg in trials:
        res = train_single_cnn_trial(
            trial_cfg=trial_cfg,
            device=device,
            max_epochs=4,
            patience=2,
        )
        all_trial_summaries.append(res)
        val_acc = res["best_metrics"]["val_accuracy"]

        # Checkpoint per trial
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        trial_ckpt = MODELS_DIR / f"cnn_{res['exp_id']}.pt"
        torch.save({
            "exp_id": res["exp_id"],
            "arch": res["arch"],
            "best_epoch": res["best_epoch"],
            "model_state_dict": res["best_state_dict"],
            "best_metrics": res["best_metrics"],
            "history": res["history"],
        }, trial_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_trial_result = res

    print("\n" + "=" * 75)
    print(f"[STAGE 1 SELECTION] Superior Architecture: {best_trial_result['exp_id']} ({best_val_acc*100:.2f}% Val Acc)")
    print("=" * 75)

    # Stage 2: Train the superior architecture for additional 6 epochs to reach full convergence
    top_cfg = best_trial_result["trial_cfg"].copy()
    top_cfg["exp_id"] = "CNN_Final_Optimized_Converged"
    top_cfg["desc"] = f"Converged Training of Superior Architecture ({best_trial_result['exp_id']})"
    print(f"\nTraining Top Architecture to Full Convergence (8 epochs with early stopping)...")

    converged_res = train_single_cnn_trial(
        trial_cfg=top_cfg,
        device=device,
        max_epochs=8,
        patience=3,
    )
    all_trial_summaries.append(converged_res)

    final_best_res = converged_res if converged_res["best_metrics"]["val_accuracy"] >= best_val_acc else best_trial_result
    final_best_acc = final_best_res["best_metrics"]["val_accuracy"]

    # Save final optimized checkpoint
    final_ckpt_path = MODELS_DIR / "cnn_optimized_best.pt"
    torch.save({
        "exp_id": final_best_res["exp_id"],
        "arch": final_best_res["arch"],
        "best_epoch": final_best_res["best_epoch"],
        "model_state_dict": final_best_res["best_state_dict"],
        "best_metrics": final_best_res["best_metrics"],
        "history": final_best_res["history"],
    }, final_ckpt_path)
    print(f"\n[FINAL OPTIMIZED CNN SAVED] {final_best_res['exp_id']} (Val Acc: {final_best_acc*100:.2f}%) -> {final_ckpt_path}")

    # Save summary table of all CNN trials
    df_cnn_summary = pd.DataFrame([
        {
            "Experiment_ID": r["exp_id"],
            "Architecture": r["arch"],
            "Best_Epoch": r["best_epoch"],
            "Train_Accuracy": round(r["best_metrics"]["train_accuracy"], 5),
            "Val_Accuracy": round(r["best_metrics"]["val_accuracy"], 5),
            "Val_Precision": round(r["best_metrics"]["val_precision"], 5),
            "Val_Recall": round(r["best_metrics"]["val_recall"], 5),
            "Val_F1": round(r["best_metrics"]["val_f1"], 5),
            "Val_ROC_AUC": round(r["best_metrics"]["val_auc"], 5),
            "Total_Time_Sec": round(r["total_time"], 1),
        }
        for r in all_trial_summaries
    ])
    df_cnn_summary.to_csv(TABLES_DIR / "cnn_phase3_experiments.csv", index=False)
    print(f"[SAVED] CNN Phase 3 experiments table saved to {TABLES_DIR / 'cnn_phase3_experiments.csv'}")


if __name__ == "__main__":
    run_cnn_optimization()
