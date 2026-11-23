"""
Handcrafted Feature Extraction for FakeLens Classical Models (KNN, Random Forest, MLP).

Reuses the 4 baseline methods:
1. RGB Color Histograms (32 bins per channel = 96 dims, normalized)
2. Local Binary Patterns (LBP) Texture (uniform LBP, P=8, R=1 -> 10 bins, normalized)
3. 2D Fast Fourier Transform (FFT) Frequency Statistics (radial energy distribution across frequency rings = 16 dims)
4. Histogram of Oriented Gradients (HOG) Edge Features (8 orientations, 16x16 pixels per cell, 1x1 cells per block = 1,568 dims)

All feature extractors return fixed-length 1D numpy vectors.
Total combined feature dimension: 96 + 10 + 16 + 1568 = 1,690 dimensions.
"""

from pathlib import Path
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from PIL import Image

from backend.preprocessing.image_transforms import load_image_array


def extract_color_histogram(img_rgb: np.ndarray, bins_per_channel: int = 32) -> np.ndarray:
    """
    Computes normalized 1D color histograms for R, G, and B channels.
    Output size: 3 * bins_per_channel (default: 96).
    """
    hist_r = cv2.calcHist([img_rgb], [0], None, [bins_per_channel], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [bins_per_channel], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [bins_per_channel], [0, 256]).flatten()

    # Normalize each channel histogram
    hist_r = hist_r / (hist_r.sum() + 1e-7)
    hist_g = hist_g / (hist_g.sum() + 1e-7)
    hist_b = hist_b / (hist_b.sum() + 1e-7)

    return np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)


def extract_lbp_texture(img_rgb: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    """
    Computes rotation-invariant uniform Local Binary Pattern (LBP) histogram on grayscale image.
    With P=8, uniform LBP yields P + 2 = 10 bins.
    Output size: strictly fixed at 10 bins [0..9].
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    num_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins), density=True)
    return hist.astype(np.float32)


def extract_fft_frequency(img_rgb: np.ndarray, num_rings: int = 16) -> np.ndarray:
    """
    Computes 2D Fast Fourier Transform (FFT) magnitude spectrum statistics.
    Measures the radial power energy across concentric frequency bands from DC to Nyquist.
    AI generative models (GANs and Diffusion) often exhibit subtle high-frequency artifacts or spectral drop-offs.
    Output size: num_rings (default: 16).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Compute 2D FFT and center low frequencies
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    # Log scale energy
    log_magnitude = np.log1p(magnitude_spectrum)

    # Create radial distance map from center
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)

    # Bin radial energies into rings
    ring_bins = np.linspace(0, max_r, num_rings + 1)
    radial_energy = np.zeros(num_rings, dtype=np.float32)

    for i in range(num_rings):
        mask = (r >= ring_bins[i]) & (r < ring_bins[i + 1])
        if np.any(mask):
            radial_energy[i] = np.mean(log_magnitude[mask])

    # Normalize radial energy
    norm = np.linalg.norm(radial_energy)
    if norm > 0:
        radial_energy = radial_energy / norm

    return radial_energy.astype(np.float32)


def extract_hog_edges(img_rgb: np.ndarray) -> np.ndarray:
    """
    Computes Histogram of Oriented Gradients (HOG) to capture facial structural edge profiles.
    Parameters for 224x224 input:
    - orientations = 8
    - pixels_per_cell = (16, 16) -> 14x14 cells
    - cells_per_block = (1, 1) -> 14 * 14 * 1 * 8 = 1,568 features
    Output size: 1,568.
    """
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


def extract_all_features(img_path_or_array: str | Path | np.ndarray) -> np.ndarray:
    """
    Extracts and concatenates all 4 baseline features into a single 1D vector:
    [Color Hist (96) | LBP (10) | FFT (16) | HOG (1568)] = 1,690 dimensions.
    """
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
