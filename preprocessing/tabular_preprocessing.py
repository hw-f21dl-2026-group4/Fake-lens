from pathlib import Path
import numpy as np
from PIL import Image
from backend.features.extractors_v3 import extract_full_combined_features
from backend.preprocessing.image_transforms import load_image_array

#convert image charteristics into a numerical feature
def extract_features_from_image(img_input: Image.Image | str | Path) -> np.ndarray:
    if isinstance(img_input, (str, Path)):
        img_arr = load_image_array(img_input, target_size=(224, 224))
    else:
        if img_input.size != (224, 224):
            img_input = img_input.resize((224, 224), Image.Resampling.BILINEAR)
        img_arr = np.array(img_input.convert("RGB"), dtype=np.uint8)

    base, ext = extract_full_combined_features(img_arr)
    feats = np.concatenate([base, ext])
    assert len(feats) == 1798, f"Expected 1798 features, got {len(feats)}"
    return feats.astype(np.float32)

#return the names of the features group used during image analysis.
def get_feature_group_names() -> list[str]:
    return [
        "Color Histograms (96 dims)",
        "LBP Texture (10 dims)",
        "2D FFT Frequency Rings (16 dims)",
        "HOG Gradients (1,568 dims)",
        "Extended Texture Dynamics (24 dims)",
        "Spectral Anisotropy & HF Ratios (16 dims)",
        "Color Channel Moments (24 dims)",
        "Noise Residuals & Tenengrad (20 dims)",
        "Spatial Symmetry (24 dims)"
    ]
