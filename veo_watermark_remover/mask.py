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
    selection_strategy: str
    detected_component_count: int
    confidence: float
    coverage_ratio: float
    span_ratio: tuple[float, float]
    rectangularity: float
    alignment_scale: float | None
    alignment_offset: tuple[int, int] | None
    requested_dilation: int
    effective_dilation: int
    final_coverage_ratio: float


class MaskCalibrationError(RuntimeError):
    def __init__(self, message: str, details: dict[str, object]) -> None:
        super().__init__(message)
        self.details = details


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
    weak_threshold = (
        (hsv[..., 1] < min(255, saturation_max + 50))
        & (hsv[..., 2] > max(0, value_min - 40))
    ).astype(np.uint8)
    bounded = np.zeros((roi_height, roi_width), dtype=np.uint8)
    bounded[local_y:local_y + logo_height, local_x:local_x + logo_width] = 1
    threshold &= bounded
    weak_threshold &= bounded

    count, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, connectivity=8)
    components = sorted(range(1, count), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
    detected_component_count = len(components)
    if detected_component_count >= 3:
        selected = components[:3]
        selection_strategy = "three_largest_components"
    elif detected_component_count in {1, 2}:
        selected = components
        selection_strategy = "adaptive_merged_components"
    else:
        raise MaskCalibrationError(
            "No Veo foreground found inside the measured logo bbox",
            {"detected_component_count": 0},
        )

    candidate = np.isin(labels, selected)
    ys, xs = np.nonzero(candidate)
    pixel_count = int(candidate.sum())
    bbox_area = logo_width * logo_height
    coverage = pixel_count / max(1, bbox_area)
    span_width = int(xs.max() - xs.min() + 1)
    span_height = int(ys.max() - ys.min() + 1)
    span_width_ratio = span_width / logo_width
    span_height_ratio = span_height / logo_height
    rectangularity = pixel_count / max(1, span_width * span_height)

    component_score = 1.0 if detected_component_count == 3 else (
        0.85 if detected_component_count > 3 else 0.72
    )
    coverage_score = max(0.0, 1.0 - abs(coverage - 0.42) / 0.42)
    span_score = min(1.0, span_width_ratio / 0.75) * min(
        1.0, span_height_ratio / 0.55
    )
    shape_score = max(0.0, min(1.0, (0.85 - rectangularity) / 0.35))
    confidence = (
        0.35 * component_score
        + 0.30 * coverage_score
        + 0.20 * span_score
        + 0.15 * shape_score
    )
    minimum_confidence = 0.72 if detected_component_count < 3 else 0.62
    plausible = (
        0.06 <= coverage <= 0.70
        and span_width_ratio >= 0.55
        and span_height_ratio >= 0.35
        and rectangularity <= 0.85
        and confidence >= minimum_confidence
    )
    calibration = {
        "selection_strategy": selection_strategy,
        "detected_component_count": detected_component_count,
        "raw_mask_pixel_count": pixel_count,
        "bbox": [local_x, local_y, logo_width, logo_height],
        "coverage_ratio": round(coverage, 6),
        "span_ratio": [round(span_width_ratio, 6), round(span_height_ratio, 6)],
        "rectangularity": round(rectangularity, 6),
        "confidence": round(confidence, 6),
        "minimum_confidence": minimum_confidence,
        "alignment_scale": None,
        "alignment_offset": None,
    }
    if not plausible:
        raise MaskCalibrationError(
            "Per-video Veo mask calibration rejected low-confidence foreground: "
            f"components={detected_component_count}, pixels={pixel_count}, "
            f"coverage={coverage:.3f}, span={span_width}x{span_height}, "
            f"rectangularity={rectangularity:.3f}, confidence={confidence:.3f}",
            calibration,
        )
    raw_mask = candidate.astype(np.uint8) * 255

    # Hysteresis refinement: expansion is allowed only onto weaker neutral-white
    # evidence observed in this video's median. This captures anti-alias edges
    # without a blind dilation or template pixels becoming the final mask.
    effective_dilation = dilation
    final_selected = raw_mask > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(effective_dilation):
        neighborhood = cv2.dilate(final_selected.astype(np.uint8), kernel) > 0
        final_selected |= neighborhood & (weak_threshold > 0)
    final_mask = final_selected.astype(np.uint8) * 255

    # Hard safety bound: one dilation pixel outside the measured visible bbox,
    # never a filled rectangle or a broad corner mask.
    safety = np.zeros_like(final_mask)
    pad = max(1, effective_dilation)
    x1, y1 = max(0, local_x - pad), max(0, local_y - pad)
    x2 = min(roi_width, local_x + logo_width + pad)
    y2 = min(roi_height, local_y + logo_height + pad)
    safety[y1:y2, x1:x2] = 255
    final_mask = cv2.bitwise_and(final_mask, safety)
    final_pixel_count = int(np.count_nonzero(final_mask))
    final_coverage = final_pixel_count / max(1, bbox_area)
    if final_coverage > 0.80:
        details = dict(calibration)
        details.update(
            {
                "final_mask_pixel_count": final_pixel_count,
                "final_coverage_ratio": round(final_coverage, 6),
                "requested_dilation": dilation,
                "effective_dilation": effective_dilation,
            }
        )
        raise MaskCalibrationError(
            "Per-video Veo final mask rejected because dilation produced a broad patch: "
            f"pixels={final_pixel_count}, coverage={final_coverage:.3f}",
            details,
        )

    return MaskResult(
        raw_mask=raw_mask,
        mask=final_mask,
        median_roi=median_roi,
        component_count=len(selected),
        raw_pixel_count=int(np.count_nonzero(raw_mask)),
        final_pixel_count=final_pixel_count,
        local_logo_bbox=(local_x, local_y, logo_width, logo_height),
        selection_strategy=selection_strategy,
        detected_component_count=detected_component_count,
        confidence=round(confidence, 6),
        coverage_ratio=round(coverage, 6),
        span_ratio=(round(span_width_ratio, 6), round(span_height_ratio, 6)),
        rectangularity=round(rectangularity, 6),
        alignment_scale=None,
        alignment_offset=None,
        requested_dilation=dilation,
        effective_dilation=effective_dilation,
        final_coverage_ratio=round(final_coverage, 6),
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
    write_image(output_dir / "mask_final.png", result.mask)
    write_image(output_dir / "mask_overlay_median.png", mask_overlay(result.median_roi, result.mask))
