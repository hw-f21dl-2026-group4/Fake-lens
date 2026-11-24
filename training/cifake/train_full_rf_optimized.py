import time
import os
import psutil
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from configs.config import MODELS_DIR, RANDOM_SEED
from src.data_loader import get_traditional_ml_data
from src.experiment_tracker import log_experiment

# to train optimized random forest on full dataset
def main():
    print("=" * 80)
    print("TRAINING FULL 79,707 REGULARIZED RANDOM FOREST (PHASE 3 OPTIMIZED)")
    print("=" * 80)

    process = psutil.Process(os.getpid())
    mem_baseline = process.memory_info().rss / (1024 * 1024)

    print("Loading full 79,707 training images and validation images in flattened RGB...")
    (X_tr, y_tr), (X_v, y_v), _ = get_traditional_ml_data(
        representation="flattened_rgb",
        train_subset_size=None,
        splits_to_load=["train", "val"],
    )
    print(f"X_tr shape: {X_tr.shape}, X_v shape: {X_v.shape}")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=35,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    t0 = time.time()
    rf.fit(X_tr, y_tr)
    t_fit = time.time() - t0
    print(f"Full fit completed in {t_fit:.1f}s")

    t0 = time.time()
    y_pred = rf.predict(X_v)
    t_pred = time.time() - t0
    y_prob = rf.predict_proba(X_v)[:, 1]

    acc = accuracy_score(y_v, y_pred)
    prec = precision_score(y_v, y_pred, pos_label=1)
    rec = recall_score(y_v, y_pred, pos_label=1)
    f1 = f1_score(y_v, y_pred, pos_label=1)
    auc = roc_auc_score(y_v, y_prob)

    # Sample train accuracy
    n_eval_tr = 10000
    tr_acc = accuracy_score(y_tr[:n_eval_tr], rf.predict(X_tr[:n_eval_tr]))
    gen_gap = (tr_acc - acc) * 100

    mem_now = process.memory_info().rss / (1024 * 1024)
    peak_mem = round(max(0, mem_now - mem_baseline), 1)

    print(f"Full 79k Regularized RF -> Val Acc: {acc*100:.2f}%, F1: {f1:.4f}, AUC: {auc:.4f}, Gap: {gen_gap:.1f}%")

    log_experiment({
        "Model": "Random Forest",
        "Experiment_ID": "RF_Full_79k_Reg_Est300",
        "Architecture": "RF(n_est=300, depth=35, leaf=2, feat=sqrt)",
        "Hyperparameters": "n_est=300, depth=35, split=5, leaf=2, feat=sqrt",
        "Feature_Representation": "Flattened-RGB",
        "Train_Samples": len(X_tr),
        "Val_Samples": len(y_v),
        "Train_Accuracy": tr_acc,
        "Val_Accuracy": acc,
        "Val_Precision": prec,
        "Val_Recall": rec,
        "Val_F1": f1,
        "Val_ROC_AUC": auc,
        "Train_Time_Sec": t_fit,
        "Infer_Time_Sec": t_pred,
        "Peak_Memory_MB": peak_mem,
        "Checkpoint_Path": "models/rf_optimized_best.joblib",
    })

    # Save best bundle
    best_bundle = {
        "model": rf,
        "config": {
            "exp_id": "RF_Full_79k_Reg_Est300",
            "n_estimators": 300,
            "max_depth": 35,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        "feature_representation": "Flattened-RGB",
        "train_samples": len(X_tr),
        "train_accuracy": tr_acc,
        "val_accuracy": acc,
        "val_f1": f1,
        "val_auc": auc,
        "gen_gap": gen_gap,
        "train_time_sec": t_fit,
        "pred_time_sec": t_pred,
    }
    save_path = MODELS_DIR / "rf_optimized_best.joblib"
    joblib.dump(best_bundle, save_path)
    print(f"--> Saved optimized Random Forest bundle to {save_path}")


if __name__ == "__main__":
    main()
