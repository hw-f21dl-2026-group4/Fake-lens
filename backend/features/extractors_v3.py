from pathlib import Path
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis

from backend.preprocessing.image_transforms import load_image_array
from backend.features.extractors import extract_all_features

# Examines detailed texture patterns and changes in pixel intensity.
# These measurements provide additional information about image structure.
def extract_texture_gradient_extended(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    lbp16 = local_binary_pattern(gray, P=16, R=2.0, method="uniform")
    n_bins = int(lbp16.max() + 1)
    hist16, _ = np.histogram(lbp16.ravel(), bins=18, range=(0, 18), density=True)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = np.var(lap)
    lap_kurt = kurtosis(lap.ravel(), fisher=True)
    lap_stats = np.array([np.log1p(max(0.0, lap_var)), lap_kurt], dtype=np.float32)

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    gm_mean = np.mean(grad_mag)
    gm_std = np.std(grad_mag)
    gm_skew = skew(grad_mag.ravel())
    gm_p95 = np.percentile(grad_mag, 95)
    sobel_stats = np.array([gm_mean, gm_std, gm_skew, gm_p95], dtype=np.float32) / 255.0

    return np.concatenate([hist16.astype(np.float32), lap_stats, sobel_stats])

# Measures how the image energy is distributed across different frequency regions.
def extract_frequency_energy_extended(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    log_mag = np.log1p(mag)

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)

    ring_bins = np.linspace(0, max_r, 9)
    radial_energy = np.zeros(8, dtype=np.float32)
    for i in range(8):
        mask = (r >= ring_bins[i]) & (r < ring_bins[i + 1])
        if np.any(mask):
            radial_energy[i] = np.mean(log_mag[mask])
    norm_r = np.linalg.norm(radial_energy)
    if norm_r > 0:
        radial_energy /= norm_r
    q1 = np.mean(log_mag[:cy, :cx])
    q2 = np.mean(log_mag[:cy, cx:])
    q3 = np.mean(log_mag[cy:, :cx])
    q4 = np.mean(log_mag[cy:, cx:])
    quadrants = np.array([q1, q2, q3, q4], dtype=np.float32)
    norm_q = np.linalg.norm(quadrants)
    if norm_q > 0:
        quadrants /= norm_q

    low_e = np.mean(log_mag[r < max_r * 0.25])
    mid_e = np.mean(log_mag[(r >= max_r * 0.25) & (r < max_r * 0.6)])
    high_e = np.mean(log_mag[r >= max_r * 0.6])
    total_e = np.mean(log_mag)

    ratio_lh = low_e / (high_e + 1e-6)
    ratio_mh = mid_e / (high_e + 1e-6)
    high_frac = high_e / (total_e + 1e-6)
    ratios = np.array([ratio_lh, ratio_mh, high_frac, total_e / 100.0], dtype=np.float32)

    return np.concatenate([radial_energy, quadrants, ratios])

# Studies the colour relationships and variations using different colour spaces
def extract_color_extended(img_rgb: np.ndarray) -> np.ndarray:
    r = img_rgb[:, :, 0].ravel().astype(np.float32)
    g = img_rgb[:, :, 1].ravel().astype(np.float32)
    b = img_rgb[:, :, 2].ravel().astype(np.float32)
    rg = np.corrcoef(r, g)[0, 1]
    rb = np.corrcoef(r, b)[0, 1]
    gb = np.corrcoef(g, b)[0, 1]
    rgb_corr = np.nan_to_num(np.array([rg, rb, gb], dtype=np.float32), nan=0.0)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_m, h_s = np.mean(hsv[:, :, 0]) / 180.0, np.std(hsv[:, :, 0]) / 180.0
    s_m, s_s = np.mean(hsv[:, :, 1]) / 255.0, np.std(hsv[:, :, 1]) / 255.0
    v_m, v_s = np.mean(hsv[:, :, 2]) / 255.0, np.std(hsv[:, :, 2]) / 255.0
    hsv_stats = np.array([h_m, h_s, s_m, s_s, v_m, v_s], dtype=np.float32)

    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr = ycrcb[:, :, 1].ravel()
    cb = ycrcb[:, :, 2].ravel()
    cr_m, cr_s, cr_sk = np.mean(cr) / 255.0, np.std(cr) / 255.0, skew(cr)
    cb_m, cb_s, cb_sk = np.mean(cb) / 255.0, np.std(cb) / 255.0, skew(cb)
    ycrcb_stats = np.nan_to_num(np.array([cr_m, cr_s, cr_sk, cb_m, cb_s, cb_sk], dtype=np.float32), nan=0.0)

    r_std, g_std, b_std = np.std(r) / 255.0, np.std(g) / 255.0, np.std(b) / 255.0
    r_sk, g_sk, b_sk = skew(r), skew(g), skew(b)
    diff_rg = np.mean(np.abs(r - g)) / 255.0
    diff_rb = np.mean(np.abs(r - b)) / 255.0
    diff_gb = np.mean(np.abs(g - b)) / 255.0
    color_dyn = np.nan_to_num(np.array([r_std, g_std, b_std, r_sk, g_sk, b_sk, diff_rg, diff_rb, diff_gb], dtype=np.float32), nan=0.0)

    return np.concatenate([rgb_corr, hsv_stats, ycrcb_stats, color_dyn])

# Measures the image noise and sharpness to capture possible irregularities
def extract_noise_sharpness_extended(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    noise_res = gray - blurred

    n_var = np.var(noise_res)
    n_kurt = kurtosis(noise_res.ravel(), fisher=True)
    n_mean = np.mean(np.abs(noise_res))
    n_max = np.max(np.abs(noise_res))
    noise_stats = np.nan_to_num(np.array([np.log1p(n_var), n_kurt, n_mean / 255.0, n_max / 255.0], dtype=np.float32), nan=0.0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ten_sq = gx ** 2 + gy ** 2
    t_mean = np.mean(ten_sq)
    t_std = np.std(ten_sq)
    t_p90 = np.percentile(ten_sq, 90)
    t_p99 = np.percentile(ten_sq, 99)
    ten_stats = np.nan_to_num(np.array([np.log1p(t_mean), np.log1p(t_std), np.log1p(t_p90), np.log1p(t_p99)], dtype=np.float32), nan=0.0)

    cy, cx = h // 2, w // 2
    nv1 = np.var(noise_res[:cy, :cx])
    nv2 = np.var(noise_res[:cy, cx:])
    nv3 = np.var(noise_res[cy:, :cx])
    nv4 = np.var(noise_res[cy:, cx:])
    q_noise = np.nan_to_num(np.log1p(np.array([nv1, nv2, nv3, nv4], dtype=np.float32)), nan=0.0)

    # Residual noise Fourier power spectrum (8 radial rings)
    f_noise = np.fft.fftshift(np.fft.fft2(noise_res))
    mag_noise = np.log1p(np.abs(f_noise))
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    ring_bins = np.linspace(0, max_r, 9)
    res_rings = np.zeros(8, dtype=np.float32)
    for i in range(8):
        mask = (r >= ring_bins[i]) & (r < ring_bins[i + 1])
        if np.any(mask):
            res_rings[i] = np.mean(mag_noise[mask])
    norm_rr = np.linalg.norm(res_rings)
    if norm_rr > 0:
        res_rings /= norm_rr

    return np.concatenate([noise_stats, ten_stats, q_noise, res_rings])

# Compares different regions of the image to check whether
def extract_spatial_consistency_extended(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    y1, y2 = int(0.25 * h), int(0.75 * h)
    x1, x2 = int(0.25 * w), int(0.75 * w)

    center = gray[y1:y2, x1:x2]
    mask_periphery = np.ones((h, w), dtype=bool)
    mask_periphery[y1:y2, x1:x2] = False
    periphery = gray[mask_periphery]

    c_mean, c_std = np.mean(center) / 255.0, np.std(center) / 255.0
    p_mean, p_std = np.mean(periphery) / 255.0, np.std(periphery) / 255.0
    diff_mean = (c_mean - p_mean)
    diff_std = (c_std - p_std)

    c_lap = np.var(cv2.Laplacian(center, cv2.CV_32F))
    p_lap = np.var(cv2.Laplacian(gray, cv2.CV_32F)[mask_periphery])
    lap_ratio = np.log1p(c_lap) / (np.log1p(p_lap) + 1e-6)

    c_rgb = img_rgb[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32) / 255.0
    p_rgb = img_rgb[mask_periphery].reshape(-1, 3).astype(np.float32) / 255.0
    c_rgb_m = np.mean(c_rgb, axis=0)
    p_rgb_m = np.mean(p_rgb, axis=0)
    c_rgb_diff = c_rgb_m - p_rgb_m
    norm_c_diff = np.linalg.norm(c_rgb_diff)
    cp_stats = np.array([c_mean, c_std, p_mean, p_std, diff_mean, diff_std, lap_ratio, c_rgb_diff[0], c_rgb_diff[1], c_rgb_diff[2], norm_c_diff, np.log1p(c_lap)], dtype=np.float32)

    upper = gray[:h//2, :]
    lower = gray[h//2:, :]
    u_mean, u_std = np.mean(upper) / 255.0, np.std(upper) / 255.0
    l_mean, l_std = np.mean(lower) / 255.0, np.std(lower) / 255.0
    ul_mean_diff = u_mean - l_mean
    ul_std_diff = u_std - l_std

    u_rgb = img_rgb[:h//2, :, :].reshape(-1, 3).astype(np.float32) / 255.0
    l_rgb = img_rgb[h//2:, :, :].reshape(-1, 3).astype(np.float32) / 255.0
    u_rgb_m = np.mean(u_rgb, axis=0)
    l_rgb_m = np.mean(l_rgb, axis=0)
    ul_rgb_diff = u_rgb_m - l_rgb_m

    u_lap = np.var(cv2.Laplacian(upper, cv2.CV_32F))
    l_lap = np.var(cv2.Laplacian(lower, cv2.CV_32F))
    ul_lap_diff = np.log1p(u_lap) - np.log1p(l_lap)

    ul_stats = np.array([u_mean, u_std, l_mean, l_std, ul_mean_diff, ul_std_diff, ul_rgb_diff[0], ul_rgb_diff[1], ul_rgb_diff[2], np.linalg.norm(ul_rgb_diff), np.log1p(u_lap), ul_lap_diff], dtype=np.float32)

    return np.concatenate([cp_stats, ul_stats])

# Brings together the additional forensic measurements into one feature vector
def extract_extended_forensic_features(img_path_or_array: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(img_path_or_array, (str, Path)):
        img_rgb = load_image_array(img_path_or_array, target_size=(224, 224))
    else:
        img_rgb = img_path_or_array

    f_text = extract_texture_gradient_extended(img_rgb)
    f_freq = extract_frequency_energy_extended(img_rgb)
    f_col = extract_color_extended(img_rgb)
    f_noise = extract_noise_sharpness_extended(img_rgb)
    f_spat = extract_spatial_consistency_extended(img_rgb)

    ext_vector = np.concatenate([f_text, f_freq, f_col, f_noise, f_spat]).astype(np.float32)
    return ext_vector

# Extract both the original and extended features from the same image.
# Return them separately for further processing
def extract_full_combined_features(img_path_or_array: str | Path | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(img_path_or_array, (str, Path)):
        img_rgb = load_image_array(img_path_or_array, target_size=(224, 224))
    else:
        img_rgb = img_path_or_array

    baseline = extract_all_features(img_rgb)
    forensic = extract_extended_forensic_features(img_rgb)
    extended = np.concatenate([baseline, forensic]).astype(np.float32)
    return baseline.astype(np.float32), extended
