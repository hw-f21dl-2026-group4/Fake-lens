# classical model initializers for tabular feature vectors

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# knn classifier configured for forensic features
def get_knn_model(n_neighbors: int = 5, metric: str = "euclidean") -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights="distance",
        n_jobs=-1
    )


# random forest ensemble classifier
def get_random_forest_model(n_estimators: int = 150, max_depth: int = 25, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )


# multilayer perceptron classifier
def get_mlp_model(hidden_layer_sizes: tuple = (256, 128, 64), max_iter: int = 100, random_state: int = 42) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=random_state,
        max_iter=max_iter
    )
