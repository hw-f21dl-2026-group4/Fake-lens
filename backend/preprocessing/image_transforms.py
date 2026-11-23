"""
Image preprocessing and transformation pipelines for FakeLens.
Supports:
- Classical feature extraction (RGB 224x224, uint8 array)
- PyTorch CNN image loading:
  - Training: Light augmentation (RandomHorizontalFlip, mild RandomRotation +-10 deg), resize to 224x224, ToTensor, Normalize
  - Validation/Testing: Resize to 224x224, ToTensor, Normalize (NO augmentation, strictly deterministic)
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
    """
    Loads an image file, safely converts it to standard 3-channel RGB,
    and resizes it consistently to the target dimensions using bilinear interpolation.
    Avoids destructive filtering or heavy compression.
    """
    img = Image.open(path).convert("RGB")
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)
    return img


def load_image_array(path: str | Path, target_size=(224, 224)) -> np.ndarray:
    """
    Returns the loaded RGB image as a uint8 numpy array with shape (H, W, 3).
    Used for handcrafted feature extraction (Histograms, LBP, FFT, HOG).
    """
    img = load_image_rgb(path, target_size=target_size)
    return np.array(img, dtype=np.uint8)


def get_cnn_transforms(split: str = "train", target_size: tuple = (224, 224)):
    """
    Returns torchvision transforms for the specified split.
    - 'train': Light augmentation (horizontal flip, subtle rotation), normalization.
    - 'val' / 'test': Strictly deterministic resize and normalization (NO augmentation).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch/torchvision is not yet installed in this environment.")

    if split == "train":
        return transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
