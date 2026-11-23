"""
PyTorch Dataset and DataLoaders for FakeLens CNN training and evaluation.
Features:
- Reads directly from manifest CSVs (train_manifest.csv, val_manifest.csv, test_manifest.csv).
- Applies split-aware transforms (light augmentation for train, deterministic for val/test).
- Preserves both binary_label (0=REAL, 1=FAKE) and source_class (for multiclass classification).
- Configurable batch_size (defaults to 16 for RTX 3050 6GB VRAM).
- Multiprocessing num_workers optimized for Windows (default 0 or 2).
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object
    DataLoader = object

from backend.preprocessing.image_transforms import get_cnn_transforms, load_image_rgb

# Mapping dictionaries
BINARY_LABEL_MAP = {"REAL": 0, "FAKE": 1}
SOURCE_CLASS_MAP = {"wiki": 0, "text2img": 1, "inpainting": 2, "insight": 3}
SOURCE_CLASS_NAMES = ["wiki", "text2img", "inpainting", "insight"]


class DeepFakeFaceDataset(Dataset):
    """
    PyTorch Dataset for DeepFakeFace subset images.
    """
    def __init__(self, manifest_csv: str | Path, root_dir: str | Path = ".", split: str = "train", transform=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to instantiate DeepFakeFaceDataset.")
        self.manifest_path = Path(manifest_csv)
        self.root_dir = Path(root_dir)
        self.split = split
        self.df = pd.read_csv(self.manifest_path)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_cnn_transforms(split=self.split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_rel_path = row["local_path"]
        full_path = self.root_dir / img_rel_path

        # Load RGB PIL image
        img = load_image_rgb(full_path, target_size=None)

        # Apply torchvision transform
        tensor_img = self.transform(img)

        binary_target = BINARY_LABEL_MAP[row["binary_label"]]
        source_target = SOURCE_CLASS_MAP[row["source_class"]]

        return {
            "image": tensor_img,
            "binary_label": torch.tensor(binary_target, dtype=torch.long),
            "source_class": torch.tensor(source_target, dtype=torch.long),
            "source_name": row["source_class"],
            "image_id": row["image_id"],
            "filename": row["filename"]
        }


# Construct PyTorch DataLoader for image training and evaluation
def get_cnn_dataloader(manifest_csv: str | Path,
                       root_dir: str | Path = ".",
                       split: str = "train",
                       batch_size: int = 16,
                       shuffle: Optional[bool] = None,
                       num_workers: int = 0) -> DataLoader:
    """
    Factory function returning a configured PyTorch DataLoader.
    Default shuffle is True for 'train', False for 'val' and 'test'.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to instantiate get_cnn_dataloader.")
    if shuffle is None:
        shuffle = (split == "train")

    ds = DeepFakeFaceDataset(manifest_csv=manifest_csv, root_dir=root_dir, split=split)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return loader
