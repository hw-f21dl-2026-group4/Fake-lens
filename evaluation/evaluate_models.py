import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from backend.api.inference_engine import UnifiedInferenceEngine, REGISTRY_PATH
#prepare a saved model checkpoints for evaluation and display its recorded experimental information
def evaluate_model(dataset: str, task: str, model_id: str, split: str = "val", allow_test: bool = False):
    #prevent accidental use of the protected test dataset 
    if split == "test" and not allow_test:
        raise PermissionError(
            "Access to protected test split is strictly blocked during cleanup. "
            "Use split='val' or provide explicit --allow_test_split authorization flag."
        )

    print(f"[FakeLens Evaluator] Initializing evaluation for: {dataset.upper()} | {task.upper()} | {model_id.upper()} | Split: {split.upper()}")
    
    #read the model registry to find the selected model details
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    model_info = registry[dataset][task][model_id]
    checkpoint_path = model_info["checkpoint_path"]

    #stop the process of the required model checkpoint is missing
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Model Family: {model_info.get('family')} ({model_info.get('type')})")
    print(f"  Classes ({model_info.get('num_classes')}): {model_info.get('class_order')}")
    print(f"  Recorded Historical Val Accuracy: {model_info.get('val_accuracy')}")
    print(f"  Recorded Historical Val Macro F1: {model_info.get('val_macro_f1')}")

    print("\n[FakeLens Evaluator] Checkpoint loaded and ready for deterministic frozen-model evaluation.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeLens Unified Model Evaluator")
    parser.add_argument("--dataset", type=str, choices=["deepfakeface", "artifact"], default="deepfakeface")
    parser.add_argument("--task", type=str, choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--allow_test_split", action="store_true", help="Explicit authorization to access protected test split")
    args = parser.parse_args()

    evaluate_model(args.dataset, args.task, args.model, split=args.split, allow_test=args.allow_test_split)
