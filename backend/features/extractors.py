from pathlib import Path
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from PIL import Image

from backend.preprocessing.image_transforms import load_image_array

# measure how the different color are distributed throughtout the image
def extract_color_histogram(img_rgb: np.ndarray, bins_per_channel: int = 32) -> np.ndarray:
    hist_r = cv2.calcHist([img_rgb], [0], None, [bins_per_channel], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [bins_per_channel], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [bins_per_channel], [0, 256]).flatten()

    hist_r = hist_r / (hist_r.sum() + 1e-7)
    hist_g = hist_g / (hist_g.sum() + 1e-7)
    hist_b = hist_b / (hist_b.sum() + 1e-7)

    return np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)

#capture local texture patterns by comparing neighbouting pixels
def extract_lbp_texture(img_rgb: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    num_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins), density=True)
    return hist.astype(np.float32)

# Details that may not be visible directly in the original image
def extract_fft_frequency(img_rgb: np.ndarray, num_rings: int = 16) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    log_magnitude = np.log1p(magnitude_spectrum)

    # Create radial distance map from center
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)

    ring_bins = np.linspace(0, max_r, num_rings + 1)
    radial_energy = np.zeros(num_rings, dtype=np.float32)

    for i in range(num_rings):
        mask = (r >= ring_bins[i]) & (r < ring_bins[i + 1])
        if np.any(mask):
            radial_energy[i] = np.mean(log_magnitude[mask])

    norm = np.linalg.norm(radial_energy)
    if norm > 0:
        radial_energy = radial_energy / norm

    return radial_energy.astype(np.float32)

# extract information about the edges andtheir directions within the image
def extract_hog_edges(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hog_feats = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )
    return hog_feats.astype(np.float32)

#combine color,texture,frequency and edge features into a single feature vector
def extract_all_features(img_path_or_array: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(img_path_or_array, (str, Path)):
        img_rgb = load_image_array(img_path_or_array, target_size=(224, 224))
    else:
        img_rgb = img_path_or_array

    c_hist = extract_color_histogram(img_rgb)
    lbp = extract_lbp_texture(img_rgb)
    fft = extract_fft_frequency(img_rgb)
    hog_f = extract_hog_edges(img_rgb)

    feat_vector = np.concatenate([c_hist, lbp, fft, hog_f])
    return feat_vector
