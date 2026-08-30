from __future__ import annotations

import cv2
import numpy as np


METHODS = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}


def inpaint_roi(roi: np.ndarray, mask: np.ndarray, method: str, radius: float = 3.0) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"Unknown inpaint method: {method}")
    if radius <= 0:
        raise ValueError("Inpaint radius must be positive")
    return cv2.inpaint(roi, mask, radius, METHODS[method])

