import argparse
from sklearn.ensemble import RandomForestClassifier

#select the random forest settings recorded for the given dataset and task
def get_historical_rf_params(dataset: str, task: str) -> dict:
    ds = dataset.lower()
    t = task.lower()
    if ds == "artifact":
        return {
            "n_estimators": 250,
            "max_depth": None,
            "min_samples_leaf": 4,
            "random_state": 42,
            "n_jobs": -1
        }
    else:  
        if t == "binary":
            return {
                "n_estimators": 150,
                "max_depth": None,
                "min_samples_leaf": 6,
                "random_state": 42,
                "n_jobs": -1
            }
        else:
            return {
                "n_estimators": 250,
                "max_depth": 28,
                "min_samples_leaf": 3,
                "random_state": 42,
                "n_jobs": -1
            }
#build the random forest classifier using trhe slected settings.
def create_rf_model(dataset: str = "artifact", task: str = "binary", **kwargs) -> RandomForestClassifier:
    params = get_historical_rf_params(dataset, task)
    params.update(kwargs)
    return RandomForestClassifier(**params)


def train_random_forest(dataset: str, task: str, dry_run: bool = False, **kwargs):
    params = get_historical_rf_params(dataset, task)
    params.update(kwargs)
    print(f"[Random Forest] Initializing {dataset.upper()} | {task.upper()} with params: {params}")
    model = RandomForestClassifier(**params)
    if dry_run:
        print("[Random Forest] Dry-run verification passed: RF model initialized successfully.")
        return model
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeLens Random Forest Training")
    parser.add_argument("--dataset", type=str, choices=["deepfakeface", "artifact"], default="artifact")
    parser.add_argument("--task", type=str, choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    train_random_forest(args.dataset, args.task, dry_run=args.dry_run)
