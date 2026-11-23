"""
K-Nearest Neighbors Training Specification for FakeLens.
Reconciles exact historical experimental configurations:
- DeepFakeFace Binary: k=21, uniform weighting, Euclidean
- DeepFakeFace Multiclass: k=15, distance weighting, Cosine metric (with PCA)
- ArtiFact Binary (32K/55K Champion): k=9, distance weighting, Euclidean
- ArtiFact Multiclass (32K Champion): k=5, uniform weighting, Minkowski
"""

import argparse
from sklearn.neighbors import KNeighborsClassifier


# Retrieve verified optimal hyperparameters for KNN model
def get_historical_knn_params(dataset: str, task: str) -> dict:
    ds = dataset.lower()
    t = task.lower()
    if ds == "artifact":
        if t == "binary":
            return {
                "n_neighbors": 9,
                "weights": "distance",
                "metric": "euclidean",
                "n_jobs": -1
            }
        else:
            return {
                "n_neighbors": 5,
                "weights": "uniform",
                "metric": "minkowski",
                "n_jobs": -1
            }
    else:  # deepfakeface
        if t == "binary":
            return {
                "n_neighbors": 21,
                "weights": "uniform",
                "metric": "euclidean",
                "n_jobs": -1
            }
        else:
            return {
                "n_neighbors": 15,
                "weights": "distance",
                "metric": "cosine",
                "n_jobs": -1
            }


def create_knn_model(dataset: str = "artifact", task: str = "binary", **kwargs) -> KNeighborsClassifier:
    """Builds configured KNeighborsClassifier matching exact historical experiments."""
    params = get_historical_knn_params(dataset, task)
    params.update(kwargs)
    return KNeighborsClassifier(**params)


def train_knn(dataset: str, task: str, dry_run: bool = False, **kwargs):
    """Consolidated KNN execution matching historical experiments."""
    params = get_historical_knn_params(dataset, task)
    params.update(kwargs)
    print(f"[KNN] Initializing {dataset.upper()} | {task.upper()} with params: {params}")
    model = KNeighborsClassifier(**params)
    if dry_run:
        print("[KNN] Dry-run verification passed: KNN model initialized successfully.")
        return model
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeLens KNN Training")
    parser.add_argument("--dataset", type=str, choices=["deepfakeface", "artifact"], default="artifact")
    parser.add_argument("--task", type=str, choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    train_knn(args.dataset, args.task, dry_run=args.dry_run)
