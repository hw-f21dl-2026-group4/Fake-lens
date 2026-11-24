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

#load the image in RGB format and resize it to the required dimensions
def load_image_rgb(path: str | Path, target_size=(224, 224)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)
    return img

#convert the resized image into a numerical array for furhter processing
def load_image_array(path: str | Path, target_size=(224, 224)) -> np.ndarray:
    img = load_image_rgb(path, target_size=target_size)
    return np.array(img, dtype=np.uint8)

#Apply the appropriate image tranformation depending on wheher the images are being used for training or evaluation
def get_cnn_transforms(split: str = "train", target_size: tuple = (224, 224)):
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
