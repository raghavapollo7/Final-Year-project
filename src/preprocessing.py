import cv2
import numpy as np


def apply_clahe_bgr(frame: np.ndarray) -> np.ndarray:
    """Enhance local contrast without destroying color information."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def gray_world_white_balance(frame: np.ndarray) -> np.ndarray:
    """Reduce underwater blue-green cast using a simple gray-world assumption."""
    result = frame.astype(np.float32)
    channel_means = result.reshape(-1, 3).mean(axis=0)
    global_mean = channel_means.mean()
    scale = global_mean / (channel_means + 1e-6)
    result *= scale
    return np.clip(result, 0, 255).astype(np.uint8)


def simple_dehaze(frame: np.ndarray) -> np.ndarray:
    """Lightweight dehazing approximation suitable for live demos."""
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=7)
    dehazed = cv2.addWeighted(frame, 1.45, blurred, -0.45, 0)
    return np.clip(dehazed, 0, 255).astype(np.uint8)


def sharpen(frame: np.ndarray) -> np.ndarray:
    """Recover some edge detail after blur and haze."""
    kernel = np.array([[0, -1, 0], [-1, 5.0, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(frame, -1, kernel)


def preprocess_underwater(frame: np.ndarray) -> np.ndarray:
    """Pipeline for underwater rescue imagery."""
    balanced = gray_world_white_balance(frame)
    contrast = apply_clahe_bgr(balanced)
    dehazed = simple_dehaze(contrast)
    return sharpen(dehazed)


def simulate_thermal(frame: np.ndarray) -> np.ndarray:
    """Create an infrared-style visualization for rubble demonstrations."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def preprocess_rubble(frame: np.ndarray, thermal: bool = False) -> np.ndarray:
    """Improve contrast in rubble imagery and optionally simulate thermal input."""
    denoised = cv2.bilateralFilter(frame, d=7, sigmaColor=60, sigmaSpace=60)
    contrast = apply_clahe_bgr(denoised)
    return simulate_thermal(contrast) if thermal else contrast
