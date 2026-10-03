"""Operações de cor: máscara de gramado, LAB, ΔE (CIEDE2000)."""

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


def mascara_gramado(bgr: np.ndarray, hsv_min, hsv_max) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(hsv_min, np.uint8), np.array(hsv_max, np.uint8)) > 0


def fracao_gramado(bgr: np.ndarray, hsv_min, hsv_max) -> float:
    if bgr.size == 0:
        return 0.0
    return float(mascara_gramado(bgr, hsv_min, hsv_max).mean())


def bgr_para_lab(pixels_bgr: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 BGR -> (N, 3) float LAB, com L entre 0 e 100."""
    rgb = pixels_bgr[:, ::-1].astype(np.float64) / 255.0
    return rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)


def hex_para_lab(cor_hex: str) -> np.ndarray:
    h = cor_hex.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float64) / 255.0
    return rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)


def hex_para_bgr(cor_hex: str) -> tuple[int, int, int]:
    h = cor_hex.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def delta_e(lab1, lab2) -> float:
    return float(deltaE_ciede2000(np.asarray(lab1, float), np.asarray(lab2, float)))
