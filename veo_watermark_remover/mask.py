from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .diagnostics import write_image


@dataclass(frozen=True)
class MaskResult:
    raw_mask: np.ndarray
    mask: np.ndarray
    median_roi: np.ndarray
    component_count: int
    raw_pixel_count: int
    final_pixel_count: int
    local_logo_bbox: tuple[int, int, int, int]


def collect_temporal_median(video_path: Path, roi_px: tuple[int, int, int, int]) -> tuple[np.ndarray, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    x, y, width, height = roi_px
    crops: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            crops.append(frame[y:y + height, x:x + width].copy())
    finally:
        capture.release()
    if not crops:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.median(np.stack(crops), axis=0).astype(np.uint8), len(crops)


def build_shape_mask(
    median_roi: np.ndarray,
    roi_px: tuple[int, int, int, int],
    logo_bbox_px: tuple[int, int, int, int],
    dilation: int = 1,
    saturation_max: int = 120,
    value_min: int = 100,
) -> MaskResult:
    if dilation < 0 or dilation > 4:
        raise ValueError("Mask dilation must be between 0 and 4 pixels")
    roi_x, roi_y, roi_width, roi_height = roi_px
    logo_x, logo_y, logo_width, logo_height = logo_bbox_px
    local_x = logo_x - roi_x
    local_y = logo_y - roi_y
    if local_x < 0 or local_y < 0 or local_x + logo_width > roi_width or local_y + logo_height > roi_height:
        raise ValueError("Measured logo bbox does not fit inside the processing ROI")

    hsv = cv2.cvtColor(median_roi, cv2.COLOR_BGR2HSV)
    threshold = ((hsv[..., 1] < saturation_max) & (hsv[..., 2] > value_min)).astype(np.uint8)
    bounded = np.zeros((roi_height, roi_width), dtype=np.uint8)
    bounded[local_y:local_y + logo_height, local_x:local_x + logo_width] = 1
    threshold &= bounded

    count, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, connectivity=8)
    components = sorted(range(1, count), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
    if len(components) < 3:
        raise RuntimeError(f"Expected three Veo letter components, found {len(components)}")
    selected = components[:3]
    raw_mask = np.isin(labels, selected).astype(np.uint8) * 255

    if dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
        final_mask = cv2.dilate(raw_mask, kernel, iterations=1)
    else:
        final_mask = raw_mask.copy()

    # Hard safety bound: one dilation pixel outside the measured visible bbox,
    # never a filled rectangle or a broad corner mask.
    safety = np.zeros_like(final_mask)
    pad = max(1, dilation)
    x1, y1 = max(0, local_x - pad), max(0, local_y - pad)
    x2 = min(roi_width, local_x + logo_width + pad)
    y2 = min(roi_height, local_y + logo_height + pad)
    safety[y1:y2, x1:x2] = 255
    final_mask = cv2.bitwise_and(final_mask, safety)

    return MaskResult(
        raw_mask=raw_mask,
        mask=final_mask,
        median_roi=median_roi,
        component_count=3,
        raw_pixel_count=int(np.count_nonzero(raw_mask)),
        final_pixel_count=int(np.count_nonzero(final_mask)),
        local_logo_bbox=(local_x, local_y, logo_width, logo_height),
    )


def mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    red = np.zeros_like(image)
    red[..., 2] = 255
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(image[selected], 0.35, red[selected], 0.65, 0)
    return overlay


def save_mask_diagnostics(output_dir: Path, result: MaskResult) -> None:
    write_image(output_dir / "temporal_median_roi.png", result.median_roi)
    write_image(output_dir / "mask_raw.png", result.raw_mask)
    write_image(output_dir / "mask_dilated_1px.png", result.mask)
    write_image(output_dir / "mask_overlay_median.png", mask_overlay(result.median_roi, result.mask))

