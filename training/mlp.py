import argparse
from sklearn.neural_network import MLPClassifier

# to get recorded mlp parameters
def get_historical_mlp_params(dataset: str, task: str) -> dict:
    ds = dataset.lower()
    t = task.lower()
    if ds == "artifact":
        if t == "binary":
            return {
                "hidden_layer_sizes": (512, 256),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.001,
                "learning_rate_init": 0.001,
                "batch_size": 256,
                "max_iter": 100,
                "random_state": 42
            }
        else:
            return {
                "hidden_layer_sizes": (512, 256),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.01,
                "learning_rate_init": 0.001,
                "batch_size": 256,
                "max_iter": 100,
                "random_state": 42
            }
    else: 
        return {
            "hidden_layer_sizes": (256, 128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.01,
            "learning_rate_init": 0.001,
            "batch_size": 256,
            "max_iter": 100,
            "random_state": 42
        }


# to initialize mlp model
def create_mlp_model(dataset: str = "artifact", task: str = "binary", **kwargs) -> MLPClassifier:
    params = get_historical_mlp_params(dataset, task)
    params.update(kwargs)
    return MLPClassifier(**params)


# to train mlp on forensic features
def train_mlp(dataset: str, task: str, dry_run: bool = False, **kwargs):
    params = get_historical_mlp_params(dataset, task)
    params.update(kwargs)
    print(f"[MLP] Initializing {dataset.upper()} | {task.upper()} with params: {params}")
    model = MLPClassifier(**params)
    if dry_run:
        print("[MLP] Dry-run verification passed: MLP model initialized successfully.")
        return model
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeLens MLP Training")
    parser.add_argument("--dataset", type=str, choices=["deepfakeface", "artifact"], default="artifact")
    parser.add_argument("--task", type=str, choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    train_mlp(args.dataset, args.task, dry_run=args.dry_run)
