from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


BILATERAL_FLOW_CODE = 4
FLOW_REJECT_NO_CLEAN_DONOR = 1
FLOW_REJECT_FORWARD_BACKWARD = 2
FLOW_REJECT_CONTEXT = 3
FLOW_REJECT_CONFIDENCE = 4

FlowEstimator = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class OpticalFlowResult:
    restored_stack: np.ndarray
    donor_mask: np.ndarray
    donor_source: np.ndarray
    confidence: np.ndarray
    unresolved_stack: np.ndarray
    rejection_reason: np.ndarray
    best_flow: np.ndarray
    best_forward_backward_error: np.ndarray
    best_candidate_confidence: np.ndarray
    best_candidate_source: np.ndarray


def _sample(array: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _propagate_context_flow(flow: np.ndarray, watermark_mask: np.ndarray, radius: int = 7) -> np.ndarray:
    """Replace logo-dominated vectors with robust nearby context vectors."""
    propagated = flow.copy()
    masked_points = np.argwhere(watermark_mask > 0)
    height, width = watermark_mask.shape
    for y, x in masked_points:
        y1, y2 = max(0, y - radius), min(height, y + radius + 1)
        x1, x2 = max(0, x - radius), min(width, x + radius + 1)
        valid = watermark_mask[y1:y2, x1:x2] == 0
        candidates = flow[y1:y2, x1:x2][valid]
        if candidates.size == 0:
            propagated[y, x] = 0
            continue
        median = np.median(candidates, axis=0)
        residual = np.linalg.norm(candidates - median, axis=1)
        robust = candidates[residual <= max(0.75, float(np.median(residual)) * 2.5)]
        propagated[y, x] = np.median(robust if robust.size else candidates, axis=0)
    return propagated


def estimate_farneback_pair(
    target: np.ndarray,
    source: np.ndarray,
    watermark_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target→source and source→target flow, using only CPU OpenCV."""
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    parameters = dict(
        pyr_scale=0.5,
        levels=4,
        winsize=17,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    forward = cv2.calcOpticalFlowFarneback(target_gray, source_gray, None, **parameters)
    backward = cv2.calcOpticalFlowFarneback(source_gray, target_gray, None, **parameters)
    return (
        _propagate_context_flow(forward, watermark_mask),
        _propagate_context_flow(backward, watermark_mask),
    )


def estimate_dis_pair(
    target: np.ndarray,
    source: np.ndarray,
    watermark_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU DIS flow candidate used to compare robustness on the small ROI."""
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    def calculate(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        estimator.setUseSpatialPropagation(True)
        return estimator.calc(first, second, None)

    return (
        _propagate_context_flow(calculate(target_gray, source_gray), watermark_mask),
        _propagate_context_flow(calculate(source_gray, target_gray), watermark_mask),
    )


def _candidate_from_source(
    target: np.ndarray,
    source: np.ndarray,
    source_restored: np.ndarray,
    source_alpha_valid: np.ndarray,
    watermark_mask: np.ndarray,
    alpha_confidence: np.ndarray,
    forward: np.ndarray,
    backward: np.ndarray,
    offset: int,
    context_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = watermark_mask.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = grid_x + forward[..., 0]
    map_y = grid_y + forward[..., 1]
    in_bounds = (map_x >= 1) & (map_x <= width - 2) & (map_y >= 1) & (map_y <= height - 2)

    warped_source = _sample(source_restored, map_x, map_y)
    unresolved_source = ((watermark_mask > 0) & ~source_alpha_valid).astype(np.float32)
    unresolved_weight = _sample(unresolved_source, map_x, map_y)
    clean_donor = in_bounds & (unresolved_weight < 1e-3)

    warped_backward = _sample(backward, map_x, map_y)
    cycle_x = map_x + warped_backward[..., 0]
    cycle_y = map_y + warped_backward[..., 1]
    fb_error = np.sqrt((cycle_x - grid_x) ** 2 + (cycle_y - grid_y) ** 2)

    warped_observed = _sample(source, map_x, map_y)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warped_gray = cv2.cvtColor(warped_observed, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gradient_x = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3)
    target_gradient_y = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3)
    source_gradient_x = cv2.Sobel(warped_gray, cv2.CV_32F, 1, 0, ksize=3)
    source_gradient_y = cv2.Sobel(warped_gray, cv2.CV_32F, 0, 1, ksize=3)
    color_difference = np.mean(np.abs(target.astype(np.float32) - warped_observed), axis=2)
    gradient_difference = np.sqrt(
        (target_gradient_x - source_gradient_x) ** 2
        + (target_gradient_y - source_gradient_y) ** 2
    )
    context = ((watermark_mask == 0) & in_bounds).astype(np.float32)
    kernel = context_radius * 2 + 1
    count = cv2.boxFilter(context, cv2.CV_32F, (kernel, kernel), normalize=False)
    color_mad = cv2.boxFilter(
        color_difference * context, cv2.CV_32F, (kernel, kernel), normalize=False
    ) / np.maximum(count, 1.0)
    gradient_mad = cv2.boxFilter(
        gradient_difference * context, cv2.CV_32F, (kernel, kernel), normalize=False
    ) / np.maximum(count, 1.0)

    sampled_alpha_confidence = _sample(alpha_confidence.astype(np.float32), map_x, map_y)
    sampled_inside_mask = _sample((watermark_mask > 0).astype(np.float32), map_x, map_y)
    source_confidence = np.where(sampled_inside_mask < 1e-3, 1.0, sampled_alpha_confidence)
    confidence = (
        np.exp(-((fb_error / 0.75) ** 2))
        * np.exp(-((color_mad / 14.0) ** 2))
        * np.exp(-((gradient_mad / 28.0) ** 2))
        * np.exp(-0.14 * (abs(offset) - 1))
        * source_confidence
    ).astype(np.float32)
    context_valid = (count >= 8) & (color_mad <= 22.0) & (gradient_mad <= 42.0)
    return warped_source, confidence, clean_donor, fb_error, context_valid


def reconstruct_optical_flow_temporal(
    original_stack: np.ndarray,
    alpha_stack: np.ndarray,
    alpha_valid_stack: np.ndarray,
    watermark_mask: np.ndarray,
    alpha_confidence: np.ndarray,
    offsets: tuple[int, ...] = (-3, -2, -1, 1, 2, 3),
    context_radius: int = 5,
    forward_backward_max: float = 1.0,
    bilateral_agreement_max: float = 16.0,
    bilateral_confidence_min: float = 0.15,
    single_confidence_min: float = 0.25,
    flow_estimator: FlowEstimator = estimate_farneback_pair,
) -> OpticalFlowResult:
    if original_stack.shape != alpha_stack.shape:
        raise ValueError("Original and alpha stacks must have the same shape")
    if alpha_valid_stack.shape != original_stack.shape[:3]:
        raise ValueError("Alpha-valid stack must match frame/ROI dimensions")
    if watermark_mask.shape != original_stack.shape[1:3]:
        raise ValueError("Watermark mask must match ROI dimensions")

    frame_count, height, width, _ = original_stack.shape
    watermark = watermark_mask > 0
    restored = alpha_stack.copy()
    donor_mask = np.zeros((frame_count, height, width), dtype=np.uint8)
    donor_source = np.zeros((frame_count, height, width), dtype=np.int8)
    confidence_map = np.zeros((frame_count, height, width), dtype=np.float32)
    rejection = np.zeros((frame_count, height, width), dtype=np.uint8)
    best_flow = np.zeros((frame_count, height, width, 2), dtype=np.float32)
    best_fb_error = np.full((frame_count, height, width), np.inf, dtype=np.float32)
    best_candidate_confidence = np.zeros((frame_count, height, width), dtype=np.float32)
    best_candidate_source = np.zeros((frame_count, height, width), dtype=np.int8)
    flow_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for target_index in range(frame_count):
        target_unresolved = watermark & ~alpha_valid_stack[target_index]
        candidates: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        any_clean = np.zeros((height, width), dtype=bool)
        any_fb = np.zeros((height, width), dtype=bool)
        any_context = np.zeros((height, width), dtype=bool)
        for offset in offsets:
            source_index = target_index + offset
            if source_index < 0 or source_index >= frame_count:
                continue
            pair = (min(target_index, source_index), max(target_index, source_index))
            if pair not in flow_cache:
                low_to_high, high_to_low = flow_estimator(
                    original_stack[pair[0]], original_stack[pair[1]], watermark_mask
                )
                flow_cache[pair] = (low_to_high, high_to_low)
            low_to_high, high_to_low = flow_cache[pair]
            if target_index < source_index:
                forward, backward = low_to_high, high_to_low
            else:
                forward, backward = high_to_low, low_to_high
            value, score, clean, fb_error, context_valid = _candidate_from_source(
                original_stack[target_index], original_stack[source_index], alpha_stack[source_index],
                alpha_valid_stack[source_index], watermark_mask, alpha_confidence,
                forward, backward, offset, context_radius,
            )
            fb_valid = fb_error <= forward_backward_max
            admissible = target_unresolved & clean & fb_valid & context_valid
            candidates[offset] = (value, score, admissible, forward, fb_error, clean)
            any_clean |= target_unresolved & clean
            any_fb |= target_unresolved & clean & fb_valid
            any_context |= admissible

        for y, x in np.argwhere(target_unresolved):
            before: list[tuple[float, int]] = []
            after: list[tuple[float, int]] = []
            for offset, (_, score, admissible, _, _, _) in candidates.items():
                if admissible[y, x]:
                    if score[y, x] > best_candidate_confidence[target_index, y, x]:
                        best_candidate_confidence[target_index, y, x] = score[y, x]
                        best_candidate_source[target_index, y, x] = offset
                    (before if offset < 0 else after).append((float(score[y, x]), offset))
            before.sort(reverse=True)
            after.sort(reverse=True)
            accepted = False
            if before and after:
                before_score, before_offset = before[0]
                after_score, after_offset = after[0]
                before_value = candidates[before_offset][0][y, x].astype(np.float32)
                after_value = candidates[after_offset][0][y, x].astype(np.float32)
                agreement = float(np.mean(np.abs(before_value - after_value)))
                if (
                    before_score >= bilateral_confidence_min
                    and after_score >= bilateral_confidence_min
                    and agreement <= bilateral_agreement_max
                ):
                    total = before_score + after_score
                    restored[target_index, y, x] = np.rint(np.clip(
                        (before_value * before_score + after_value * after_score) / total, 0, 255
                    )).astype(np.uint8)
                    donor_source[target_index, y, x] = BILATERAL_FLOW_CODE
                    confidence_map[target_index, y, x] = min(
                        1.0, (before_score + after_score) * 0.575
                    )
                    chosen = before_offset if before_score >= after_score else after_offset
                    best_flow[target_index, y, x] = candidates[chosen][3][y, x]
                    best_fb_error[target_index, y, x] = candidates[chosen][4][y, x]
                    accepted = True
            if not accepted:
                one_sided = before + after
                if one_sided:
                    score, offset = max(one_sided)
                    if score >= single_confidence_min:
                        restored[target_index, y, x] = candidates[offset][0][y, x]
                        donor_source[target_index, y, x] = offset
                        confidence_map[target_index, y, x] = score
                        best_flow[target_index, y, x] = candidates[offset][3][y, x]
                        best_fb_error[target_index, y, x] = candidates[offset][4][y, x]
                        accepted = True
            if accepted:
                donor_mask[target_index, y, x] = 255
            elif not any_clean[y, x]:
                rejection[target_index, y, x] = FLOW_REJECT_NO_CLEAN_DONOR
            elif not any_fb[y, x]:
                rejection[target_index, y, x] = FLOW_REJECT_FORWARD_BACKWARD
            elif not any_context[y, x]:
                rejection[target_index, y, x] = FLOW_REJECT_CONTEXT
            else:
                rejection[target_index, y, x] = FLOW_REJECT_CONFIDENCE

    unresolved = np.broadcast_to(watermark, alpha_valid_stack.shape).copy()
    unresolved &= ~alpha_valid_stack
    unresolved &= donor_mask == 0
    return OpticalFlowResult(
        restored_stack=restored,
        donor_mask=donor_mask,
        donor_source=donor_source,
        confidence=confidence_map,
        unresolved_stack=unresolved.astype(np.uint8) * 255,
        rejection_reason=rejection,
        best_flow=best_flow,
        best_forward_backward_error=best_fb_error,
        best_candidate_confidence=best_candidate_confidence,
        best_candidate_source=best_candidate_source,
    )


def visualize_flow(flow: np.ndarray, selected: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((*selected.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2.0, 180).astype(np.uint8)
    hsv[..., 1] = 255
    scale = float(np.percentile(magnitude[selected], 95)) if np.any(selected) else 0.0
    hsv[..., 2] = np.clip(magnitude / max(scale, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[~selected] = 0
    return bgr
