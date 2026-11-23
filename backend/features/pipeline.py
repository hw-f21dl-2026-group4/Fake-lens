"""
Feature pipeline and caching utilities for classical ML models (KNN, Random Forest, MLP).

Workflow:
1. Iterates over manifest splits (train, val, test).
2. Extracts the 1,690-dimensional feature vector per image.
3. Caches raw features to `results/features_cache/` in compressed .npz / parquet format.
4. Fits StandardScaler ONLY on the training split, and transforms validation and test splits without leakage.
5. Saves the fitted pipeline via joblib so inference reproduces identical scaling.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from backend.features.extractors import extract_all_features
from backend.datasets.dff_dataset import BINARY_LABEL_MAP, SOURCE_CLASS_MAP


def extract_split_features(manifest_csv: str | Path, root_dir: str | Path = ".", max_samples: int = None) -> Dict[str, Any]:
    """
    Extracts features for all images in a given manifest CSV.
    Returns a dictionary with:
    - 'X': float32 numpy array of shape (N, 1690)
    - 'y_binary': int64 array of shape (N,)
    - 'y_source': int64 array of shape (N,)
    - 'image_ids': list of strings
    """
    root = Path(root_dir)
    df = pd.read_csv(manifest_csv)
    if max_samples is not None:
        df = df.iloc[:max_samples]

    features_list = []
    y_binary_list = []
    y_source_list = []
    image_ids = []

    print(f"Extracting features for {len(df)} images from {Path(manifest_csv).name}...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Features ({Path(manifest_csv).stem})"):
        full_path = root / row["local_path"]
        feat = extract_all_features(full_path)

        features_list.append(feat)
        y_binary_list.append(BINARY_LABEL_MAP[row["binary_label"]])
        y_source_list.append(SOURCE_CLASS_MAP[row["source_class"]])
        image_ids.append(row["image_id"])

    X = np.stack(features_list).astype(np.float32)
    y_binary = np.array(y_binary_list, dtype=np.int64)
    y_source = np.array(y_source_list, dtype=np.int64)

    return {
        "X": X,
        "y_binary": y_binary,
        "y_source": y_source,
        "image_ids": image_ids
    }


def build_and_fit_scaler(X_train: np.ndarray, save_path: str | Path = None) -> Pipeline:
    """
    Fits an Imputer + StandardScaler pipeline strictly on training features.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    pipeline.fit(X_train)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        print(f"Saved fitted preprocessing pipeline to: {save_path}")

    return pipeline
