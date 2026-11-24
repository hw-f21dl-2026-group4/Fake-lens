import argparse
from sklearn.neighbors import KNeighborsClassifier

# to get recorded knn parameters
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
    else:  
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


# to initialize the knn model
def create_knn_model(dataset: str = "artifact", task: str = "binary", **kwargs) -> KNeighborsClassifier:
    params = get_historical_knn_params(dataset, task)
    params.update(kwargs)
    return KNeighborsClassifier(**params)

# to train knn on forensic features
def train_knn(dataset: str, task: str, dry_run: bool = False, **kwargs):
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
