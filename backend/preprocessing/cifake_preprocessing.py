import json
import os
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import joblib

CIFAKE_CHANNEL_MEAN = [0.47187030446185585, 0.46298588696063175, 0.4182617484016478]
CIFAKE_CHANNEL_STD = [0.23743835452750506, 0.23727779851417555, 0.2658298828347467]

CIFAKE_EVAL_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=CIFAKE_CHANNEL_MEAN, std=CIFAKE_CHANNEL_STD)
])

_PCA_100_CACHE = None

#loads the saved pca transformations used to reduce the number of features
def get_pca_100():
    global _PCA_100_CACHE
    if _PCA_100_CACHE is None:
        pca_path = os.path.join("trained_models", "CIFAKE", "binary", "pca_models.joblib")
        if not os.path.exists(pca_path):
            # Fallback to teammate directory if not yet copied
            pca_path = os.path.join("tabular dataset", "CIFAKE_Binary_Classification", "models", "pca_models.joblib")
        pcas = joblib.load(pca_path)
        _PCA_100_CACHE = pcas[100]
    return _PCA_100_CACHE


def preprocess_cifake_image(pil_img: Image.Image, model_family: str, device: torch.device = torch.device("cpu")):

    img_rgb = pil_img.convert("RGB")
    if img_rgb.size != (32, 32):
        img_rgb = img_rgb.resize((32, 32), Image.BILINEAR)

    fam = model_family.lower().strip()

    if fam in ["cnn", "custom_cnn"]:
        tensor = CIFAKE_EVAL_TRANSFORM(img_rgb).unsqueeze(0).to(device)
        return tensor

    elif fam in ["mlp", "multi-layer perceptron"]:
        tensor = CIFAKE_EVAL_TRANSFORM(img_rgb)
        tensor_flat = tensor.view(1, -1).to(device)
        return tensor_flat

    elif fam in ["random_forest", "rf"]:
        arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
        feat_vec = arr.flatten().reshape(1, -1)
        return feat_vec

    elif fam in ["knn", "k-nearest neighbors"]:
        arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
        feat_vec = arr.flatten().reshape(1, -1)
        pca = get_pca_100()
        feat_pca = pca.transform(feat_vec)
        return feat_pca

    else:
        raise ValueError(f"Unsupported model family for CIFAKE: {model_family}")
