from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


BILATERAL_DONOR_CODE = 4
REJECT_NO_CLEAN_SOURCE = 1
REJECT_CONTEXT_MISMATCH = 2
REJECT_CONFIDENCE_OR_CONSENSUS = 3


@dataclass(frozen=True)
class TemporalResult:
    restored_stack: np.ndarray
    donor_mask: np.ndarray
    donor_source: np.ndarray
    confidence: np.ndarray
    unresolved_stack: np.ndarray
    rejection_reason: np.ndarray


def _context_score(
    roi_stack: np.ndarray,
    gradient_stack: np.ndarray,
    mask: np.ndarray,
    target_frame: int,
    source_frame: int,
    y: int,
    x: int,
    radius: int,
) -> tuple[float, float, float]:
    height, width = mask.shape
    y1, y2 = max(0, y - radius), min(height, y + radius + 1)
    x1, x2 = max(0, x - radius), min(width, x + radius + 1)
    context = mask[y1:y2, x1:x2] == 0
    if np.count_nonzero(context) < 8:
        return 0.0, 255.0, 255.0
    target_patch = roi_stack[target_frame, y1:y2, x1:x2].astype(np.float32)
    source_patch = roi_stack[source_frame, y1:y2, x1:x2].astype(np.float32)
    color_mad = float(np.mean(np.abs(target_patch[context] - source_patch[context])))
    target_gradient = gradient_stack[target_frame, y1:y2, x1:x2]
    source_gradient = gradient_stack[source_frame, y1:y2, x1:x2]
    gradient_mad = float(np.mean(np.abs(target_gradient[context] - source_gradient[context])))
    distance = abs(source_frame - target_frame)
    score = float(
        np.exp(-((color_mad / 12.0) ** 2))
        * np.exp(-((gradient_mad / 18.0) ** 2))
        * np.exp(-0.18 * (distance - 1))
    )
    return score, color_mad, gradient_mad


def reconstruct_direct_temporal(
    original_stack: np.ndarray,
    alpha_stack: np.ndarray,
    alpha_valid_stack: np.ndarray,
    watermark_mask: np.ndarray,
    alpha_confidence: np.ndarray,
    offsets: tuple[int, ...] = (-3, -2, -1, 1, 2, 3),
    context_radius: int = 4,
    context_color_max: float = 18.0,
    context_gradient_max: float = 24.0,
    bilateral_agreement_max: float = 14.0,
    bilateral_confidence_min: float = 0.30,
    single_confidence_min: float = 0.58,
) -> TemporalResult:
    if original_stack.shape != alpha_stack.shape:
        raise ValueError("Original and alpha-restored stacks must have the same shape")
    if alpha_valid_stack.shape != original_stack.shape[:3]:
        raise ValueError("Alpha-valid stack must match frame/ROI dimensions")
    if watermark_mask.shape != original_stack.shape[1:3]:
        raise ValueError("Watermark mask must match ROI dimensions")

    frame_count, height, width, _ = original_stack.shape
    restored = alpha_stack.copy()
    donor_mask = np.zeros((frame_count, height, width), dtype=np.uint8)
    donor_source = np.zeros((frame_count, height, width), dtype=np.int8)
    temporal_confidence = np.zeros((frame_count, height, width), dtype=np.float32)
    rejection_reason = np.zeros((frame_count, height, width), dtype=np.uint8)
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in original_stack]).astype(np.float32)
    gradient_x = np.stack([cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3) for frame in gray])
    gradient_y = np.stack([cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3) for frame in gray])
    gradients = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    watermark = watermark_mask > 0

    for target_frame in range(frame_count):
        targets = np.argwhere(watermark & ~alpha_valid_stack[target_frame])
        for y, x in targets:
            before: list[tuple[float, int]] = []
            after: list[tuple[float, int]] = []
            clean_source_count = 0
            context_candidate_count = 0
            for offset in offsets:
                source_frame = target_frame + offset
                if source_frame < 0 or source_frame >= frame_count:
                    continue
                # A source pixel is admissible only if Experiment 2 actually
                # reconstructed this exact pixel in that exact source frame.
                if not alpha_valid_stack[source_frame, y, x]:
                    continue
                clean_source_count += 1
                context_score, color_mad, gradient_mad = _context_score(
                    original_stack, gradients, watermark_mask, target_frame,
                    source_frame, int(y), int(x), context_radius,
                )
                if color_mad > context_color_max or gradient_mad > context_gradient_max:
                    continue
                context_candidate_count += 1
                confidence = float(context_score * alpha_confidence[y, x])
                (before if offset < 0 else after).append((confidence, offset))

            before.sort(reverse=True)
            after.sort(reverse=True)
            accepted = False
            if before and after:
                before_confidence, before_offset = before[0]
                after_confidence, after_offset = after[0]
                before_value = alpha_stack[target_frame + before_offset, y, x].astype(np.float32)
                after_value = alpha_stack[target_frame + after_offset, y, x].astype(np.float32)
                agreement = float(np.mean(np.abs(before_value - after_value)))
                if (
                    before_confidence >= bilateral_confidence_min
                    and after_confidence >= bilateral_confidence_min
                    and agreement <= bilateral_agreement_max
                ):
                    total = before_confidence + after_confidence
                    restored[target_frame, y, x] = np.clip(
                        (before_value * before_confidence + after_value * after_confidence) / total,
                        0, 255,
                    ).astype(np.uint8)
                    donor_source[target_frame, y, x] = BILATERAL_DONOR_CODE
                    temporal_confidence[target_frame, y, x] = min(
                        1.0, ((before_confidence + after_confidence) / 2.0) * 1.15
                    )
                    accepted = True
            if not accepted:
                candidates = before + after
                if candidates:
                    confidence, offset = max(candidates)
                    if confidence >= single_confidence_min:
                        restored[target_frame, y, x] = alpha_stack[target_frame + offset, y, x]
                        donor_source[target_frame, y, x] = offset
                        temporal_confidence[target_frame, y, x] = confidence
                        accepted = True
            if accepted:
                donor_mask[target_frame, y, x] = 255
            elif clean_source_count == 0:
                rejection_reason[target_frame, y, x] = REJECT_NO_CLEAN_SOURCE
            elif context_candidate_count == 0:
                rejection_reason[target_frame, y, x] = REJECT_CONTEXT_MISMATCH
            else:
                rejection_reason[target_frame, y, x] = REJECT_CONFIDENCE_OR_CONSENSUS

    unresolved = np.broadcast_to(watermark, alpha_valid_stack.shape).copy()
    unresolved &= ~alpha_valid_stack
    unresolved &= donor_mask == 0
    return TemporalResult(
        restored_stack=restored,
        donor_mask=donor_mask,
        donor_source=donor_source,
        confidence=temporal_confidence,
        unresolved_stack=unresolved.astype(np.uint8) * 255,
        rejection_reason=rejection_reason,
    )


def visualize_donor_source(source_map: np.ndarray) -> np.ndarray:
    colors = {
        -3: (255, 80, 0), -2: (255, 150, 0), -1: (255, 230, 0),
        1: (0, 230, 255), 2: (0, 150, 255), 3: (0, 80, 255),
        BILATERAL_DONOR_CODE: (0, 255, 0),
    }
    output = np.zeros((*source_map.shape, 3), dtype=np.uint8)
    for code, color in colors.items():
        output[source_map == code] = color
    return output
