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
from src.models_architectures_phase3 import CIFAKEMLP_A, CIFAKEMLP_B, CIFAKEMLP_C
from src.data_loader import get_pytorch_dataloaders
from src.experiment_tracker import log_experiment
from src.validation import set_seed

#train one mlp configuration and record iots perfomarnce
def train_single_mlp_trial(
    trial_cfg: Dict[str, Any],
    train_loader,
    val_loader,
    device,
    max_epochs: int = 10,
    patience: int = 3,
) -> Dict[str, Any]:
    exp_id = trial_cfg["exp_id"]
    print("\n" + "=" * 75)
    print(f"RUNNING MLP TRIAL: {exp_id} ({trial_cfg['desc']})")
    print("=" * 75)
    set_seed(RANDOM_SEED)

    process = psutil.Process(os.getpid())
    mem_baseline = process.memory_info().rss / (1024 * 1024)

    # Instantiate model
    arch_type = trial_cfg["arch"]
    dropout = trial_cfg.get("dropout", 0.3)
    if arch_type == "A":
        model = CIFAKEMLP_A(dropout_rate=dropout).to(device)
    elif arch_type == "B":
        model = CIFAKEMLP_B(dropout_rate=dropout).to(device)
    elif arch_type == "C":
        model = CIFAKEMLP_C(dropout_rate=dropout).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch_type}")

    criterion = nn.BCEWithLogitsLoss()
    lr = trial_cfg.get("lr", 0.001)
    weight_decay = trial_cfg.get("weight_decay", 1e-4)

    if trial_cfg.get("optimizer", "adamw") == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched_type = trial_cfg.get("scheduler", "plateau")
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
    log_experiment({
        "Model": "MLP",
        "Experiment_ID": exp_id,
        "Architecture": f"CIFAKEMLP_{arch_type}",
        "Hyperparameters": f"opt={trial_cfg['optimizer']}, lr={lr}, wd={weight_decay}, drop={dropout}, sched={sched_type}",
        "Feature_Representation": "Flattened-RGB (3072)",
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
        "Checkpoint_Path": "models/mlp_optimized_best.pt",
    })

    return {
        "exp_id": exp_id,
        "arch": arch_type,
        "best_epoch": best_epoch,
        "best_state_dict": best_state_dict,
        "best_metrics": best_metrics,
        "history": history,
        "total_time": total_time,
    }


# Execute hyperparameter search for multilayer perceptron
def run_mlp_optimization():
    print("=" * 80)
    print("PHASE 3: MLP CONTROLLED OPTIMIZATION EXPERIMENTS")
    print("=" * 80)
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading PyTorch DataLoaders for MLP (Train: 79,707, Val: 19,619)...")
    train_loader, val_loader, _ = get_pytorch_dataloaders(
        batch_size=128,
        augment_train=False,
        model_type="mlp",
    )

    trials = [
        {
            "exp_id": "MLP_ArchA_AdamW_Cosine",
            "arch": "A",
            "optimizer": "adamw",
            "lr": 0.001,
            "weight_decay": 1e-4,
            "dropout": 0.3,
            "scheduler": "cosine",
            "desc": "Arch A: 3072->1024->512->128 with AdamW + Cosine",
        },
        {
            "exp_id": "MLP_ArchB_Wider_AdamW",
            "arch": "B",
            "optimizer": "adamw",
            "lr": 0.001,
            "weight_decay": 1e-4,
            "dropout": 0.35,
            "scheduler": "plateau",
            "desc": "Arch B: Wider 2048->1024->256 with AdamW + Plateau",
        },
        {
            "exp_id": "MLP_ArchC_Deeper_AdamW",
            "arch": "C",
            "optimizer": "adamw",
            "lr": 0.001,
            "weight_decay": 1e-4,
            "dropout": 0.25,
            "scheduler": "cosine",
            "desc": "Arch C: Deeper 4-Layer Hierarchy with AdamW + Cosine",
        },
        {
            "exp_id": "MLP_ArchA_LR0005_Plateau",
            "arch": "A",
            "optimizer": "adamw",
            "lr": 0.0005,
            "weight_decay": 1e-4,
            "dropout": 0.3,
            "scheduler": "plateau",
            "desc": "Arch A: Fine-grained lr=0.0005 + Plateau",
        },
    ]

    best_val_acc = -1.0
    best_trial_result = None
    all_trial_summaries = []

    for trial_cfg in trials:
        res = train_single_mlp_trial(
            trial_cfg=trial_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            max_epochs=8,
            patience=3,
        )
        all_trial_summaries.append(res)
        val_acc = res["best_metrics"]["val_accuracy"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_trial_result = res
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            ckpt_path = MODELS_DIR / "mlp_optimized_best.pt"
            torch.save({
                "exp_id": res["exp_id"],
                "arch": res["arch"],
                "best_epoch": res["best_epoch"],
                "model_state_dict": res["best_state_dict"],
                "best_metrics": res["best_metrics"],
                "history": res["history"],
            }, ckpt_path)
            print(f"--> Saved new best MLP checkpoint (Val Acc: {best_val_acc*100:.2f}%) to {ckpt_path}")

    df_mlp_summary = pd.DataFrame([
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
    df_mlp_summary.to_csv(TABLES_DIR / "mlp_phase3_experiments.csv", index=False)
    print(f"\n[MLP OPTIMIZATION COMPLETE] Saved summary to {TABLES_DIR / 'mlp_phase3_experiments.csv'}")
    print(f"Top MLP: {best_trial_result['exp_id']} ({best_val_acc*100:.2f}% Val Accuracy)")


if __name__ == "__main__":
    run_mlp_optimization()
