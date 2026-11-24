import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from backend.models.classical_models import get_knn_model, get_random_forest_model, get_mlp_model

#Compares the predicted labels woth the actual labels
def evaluate_predictions(y_true, y_pred, model_name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix (Rows=True [REAL=0, FAKE=1], Cols=Pred):")
    print(cm)

    return {
        "model": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }

#trains and compares knn, random forest and mlp using extracvted image features
def train_and_eval_classical_models():
    print("=" * 60)
    print(" FakeLens: Training & Evaluating Classical Models (KNN, RF, MLP)")
    print("=" * 60)

    root = Path(__file__).resolve().parent.parent.parent
    cache_dir = root / "results" / "features_cache"
    models_dir = root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loads Caches
    print("\n[1] Loading precomputed feature matrices...")
    train_npz = np.load(cache_dir / "train_features.npz")
    val_npz = np.load(cache_dir / "val_features.npz")
    test_npz = np.load(cache_dir / "test_features.npz")

    X_train = train_npz["X_scaled"]
    y_train = train_npz["y_binary"]

    X_val = val_npz["X_scaled"]
    y_val = val_npz["y_binary"]

    X_test = test_npz["X_scaled"]
    y_test = test_npz["y_binary"]

    print(f"Train set: {X_train.shape}, Real={np.sum(y_train==0)}, Fake={np.sum(y_train==1)}")
    print(f"Val set:   {X_val.shape}, Real={np.sum(y_val==0)}, Fake={np.sum(y_val==1)}")
    print(f"Test set:  {X_test.shape}, Real={np.sum(y_test==0)}, Fake={np.sum(y_test==1)}")

    results = []

    # 2. Trains & Tunes KNN
    print("\n[2] Training K-Nearest Neighbors (KNN)...")
    t0 = time.time()
    best_k = 5
    best_val_f1 = -1

    for k in [3, 5, 7]:
        knn_candidate = get_knn_model(n_neighbors=k)
        knn_candidate.fit(X_train, y_train)
        val_pred = knn_candidate.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="binary", zero_division=0)
        print(f"  KNN (k={k}) -> Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_k = k

    print(f"Selected best KNN: k={best_k}")
    best_knn = get_knn_model(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    knn_train_time = time.time() - t0
    joblib.dump(best_knn, models_dir / "knn_model.joblib")

    test_pred_knn = best_knn.predict(X_test)
    res_knn = evaluate_predictions(y_test, test_pred_knn, f"KNN (k={best_k})")
    res_knn["train_time_sec"] = knn_train_time
    results.append(res_knn)

    # 3. Trains & Evaluates Random Forest
    print("\n[3] Training Random Forest (RF)...")
    t0 = time.time()
    rf = get_random_forest_model(n_estimators=150, max_depth=25, random_state=42)
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    joblib.dump(rf, models_dir / "random_forest_model.joblib")
    print(f"Random Forest trained in {rf_train_time:.1f}s.")

    val_pred_rf = rf.predict(X_val)
    print(f"  RF Val F1: {f1_score(y_val, val_pred_rf, average='binary'):.4f}")

    test_pred_rf = rf.predict(X_test)
    res_rf = evaluate_predictions(y_test, test_pred_rf, "Random Forest")
    res_rf["train_time_sec"] = rf_train_time
    results.append(res_rf)

    # 4. Trains & Evaluates MLP
    print("\n[4] Training Multilayer Perceptron (MLP)...")
    t0 = time.time()
    mlp = get_mlp_model(hidden_layer_sizes=(256, 128, 64), max_iter=150, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_train_time = time.time() - t0
    joblib.dump(mlp, models_dir / "mlp_model.joblib")
    print(f"MLP trained in {mlp_train_time:.1f}s across {mlp.n_iter_} iterations.")

    val_pred_mlp = mlp.predict(X_val)
    print(f"  MLP Val F1: {f1_score(y_val, val_pred_mlp, average='binary'):.4f}")

    test_pred_mlp = mlp.predict(X_test)
    res_mlp = evaluate_predictions(y_test, test_pred_mlp, "MLP")
    res_mlp["train_time_sec"] = mlp_train_time
    results.append(res_mlp)

    summary_df = pd.DataFrame([{
        "Model": r["model"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1-Score": r["f1"],
        "Train Time (s)": round(r["train_time_sec"], 2)
    } for r in results])

    summary_csv = root / "results" / "classical_models_evaluation.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved metrics summary to: {summary_csv}")
    print(summary_df)


if __name__ == "__main__":
    train_and_eval_classical_models()
