"""
CIFAKE Binary Image Classification Project
Step 5: K-Nearest Neighbors (KNN) Baseline Pipeline
"""

import sys
import time
import os
import psutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    MODELS_DIR,
    TABLES_DIR,
    SPLITS_DIR,
    RANDOM_SEED,
)
from src.data_loader import get_traditional_ml_data
from src.validation import set_seed


# Train K-Nearest Neighbors classifier on PCA feature vectors
def train_knn_baseline(
    pca_components: int = 100,
    train_subset_size: int = 20000,
    k_candidates: List[int] = [3, 5, 7, 11, 15],
    weights_candidates: List[str] = ["uniform", "distance"],
) -> Dict[str, Any]:
    print("=" * 75)
    print("STEP 5: K-NEAREST NEIGHBORS (KNN) BASELINE")
    print("=" * 75)
    set_seed(RANDOM_SEED)

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    print(f"Loading data with PCA representation (Components: {pca_components})...")
    print(f"Training subset size: {train_subset_size:,} images (Stratified, seed=42)")
    print(f"Validation set size:  19,619 images (Full clean validation partition)")

    # Extract PCA features for Train and Validation only (Test remains locked)
    (X_train, y_train), (X_val, y_val), _ = get_traditional_ml_data(
        representation="pca",
        pca_components=pca_components,
        train_subset_size=train_subset_size,
        splits_to_load=["train", "val"],
    )

    print(f"Feature matrix shapes: X_train={X_train.shape}, X_val={X_val.shape}")

    results = []
    best_val_f1 = -1.0
    best_config = None
    best_model = None

    print("\nEvaluating Candidate Configurations on Validation Data:\n")

    for k in k_candidates:
        for w in weights_candidates:
            t0 = time.time()
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights=w,
                algorithm="auto",
                metric="euclidean",
                n_jobs=-1,
            )
            knn.fit(X_train, y_train)
            train_time = time.time() - t0

            t_pred0 = time.time()
            y_val_pred = knn.predict(X_val)
            pred_time = time.time() - t_pred0

            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average="macro")

            mem_now = process.memory_info().rss / (1024 * 1024)
            delta_mem = max(0, mem_now - mem_before)

            print(
                f"  K={k:2d} | Weights={w:8s} | "
                f"Val Acc: {val_acc*100:.2f}% | Macro F1: {val_f1:.4f} | "
                f"Fit Time: {train_time:.2f}s | Pred Time: {pred_time:.2f}s | "
                f"Mem: {delta_mem:.1f} MB"
            )

            row = {
                "k": k,
                "weights": w,
                "pca_components": pca_components,
                "train_samples": train_subset_size,
                "val_samples": len(y_val),
                "val_accuracy": round(val_acc, 5),
                "val_macro_f1": round(val_f1, 5),
                "train_time_sec": round(train_time, 3),
                "pred_time_sec": round(pred_time, 3),
                "memory_mb": round(delta_mem, 1),
            }
            results.append(row)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_config = row
                best_model = knn

    print(f"\n[SELECTION] Best KNN Configuration: K={best_config['k']}, Weights='{best_config['weights']}'")
    print(f"            Validation Accuracy: {best_config['val_accuracy']*100:.2f}%, Macro F1: {best_config['val_macro_f1']:.4f}")

    # Compute training subset accuracy for fair comparison
    print("Evaluating best KNN model on training subset...")
    t_train_eval = time.time()
    y_train_pred = best_model.predict(X_train)
    knn_train_acc = accuracy_score(y_train, y_train_pred)
    print(f"  Training Accuracy (N={len(y_train):,}): {knn_train_acc*100:.2f}% (Eval: {time.time()-t_train_eval:.2f}s)")

    # Save best model bundle
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    knn_bundle = {
        "model": best_model,
        "config": best_config,
        "pca_components": pca_components,
        "train_subset_size": train_subset_size,
        "train_accuracy": knn_train_acc,
    }
    joblib.dump(knn_bundle, MODELS_DIR / "knn_best.joblib")
    print(f"[SAVED] Best KNN model bundle written to {MODELS_DIR / 'knn_best.joblib'}")

    # Save results table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(TABLES_DIR / "knn_hyperparameter_tuning.csv", index=False)
    print(f"[SAVED] KNN tuning table written to {TABLES_DIR / 'knn_hyperparameter_tuning.csv'}")

    return {
        "best_config": best_config,
        "best_model": best_model,
        "results": results,
    }


if __name__ == "__main__":
    train_knn_baseline()
