"""
Unified Preprocessing Module for FakeLens.
Preserves the exact preprocessing profiles used in historical experiments:
- CNN & Classical: Bilinear resize to 224x224, standard ImageNet normalization.
- Augmentations: Random horizontal flip (p=0.5) only when training deep neural networks.
  Strictly deterministic resizing and normalization for validation/test/inference.
- Tabular feature extraction: 1,798-dimensional handcrafted representations with train-only scaling.
"""

from pathlib import Path
from PIL import Image
import numpy as np

try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image_rgb(path: str | Path, target_size=(224, 224)) -> Image.Image:
    """Safely loads an image from disk, converts to 3-channel RGB, and resizes using bilinear interpolation."""
    img = Image.open(path).convert("RGB")
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)
    return img


def load_image_array(path: str | Path, target_size=(224, 224)) -> np.ndarray:
    """Returns image as uint8 numpy array (H, W, 3)."""
    img = load_image_rgb(path, target_size=target_size)
    return np.array(img, dtype=np.uint8)


def get_image_transforms(dataset: str = "deepfakeface", model_family: str = "efficientnet_b0", split: str = "val"):
    """
    Returns torchvision transforms corresponding exactly to historical training & validation pipelines.
    - Training: Resize to 224x224, RandomHorizontalFlip(p=0.5), ToTensor(), Normalize(ImageNet).
    - Validation/Testing/Inference: Resize to 224x224, ToTensor(), Normalize(ImageNet) [strictly deterministic].
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch and torchvision are required for image transforms.")

    target_size = (224, 224)

    if split == "train":
        transform_list = [
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
    else:
        transform_list = [
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]

    return transforms.Compose(transform_list)


def preprocess_image_for_inference(img: Image.Image | str | Path, model_family: str = "efficientnet_b0") -> "torch.Tensor":
    """Preprocesses a PIL Image or file path into a deterministic [1, 3, 224, 224] PyTorch tensor."""
    if not isinstance(img, Image.Image):
        img = load_image_rgb(img, target_size=(224, 224))
    elif img.size != (224, 224):
        img = img.resize((224, 224), Image.Resampling.BILINEAR)

    t = get_image_transforms(model_family=model_family, split="val")
    return t(img).unsqueeze(0)
