from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AlphaModel:
    watermark_bgr: np.ndarray
    alpha: np.ndarray
    confidence: np.ndarray
    resolved_mask: np.ndarray
    transparent_mask: np.ndarray
    unresolved_mask: np.ndarray
    fit_rmse: np.ndarray
    spatial_rmse: np.ndarray
    background_dynamic_range: np.ndarray
    gamut_fraction: np.ndarray


def estimate_spatial_backgrounds(
    roi_stack: np.ndarray,
    mask: np.ndarray,
    radius: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate masked backgrounds from same-frame unmasked neighbors using local planes."""
    if roi_stack.ndim != 4 or roi_stack.shape[-1] != 3:
        raise ValueError("ROI stack must have shape [frames, height, width, 3]")
    if mask.shape != roi_stack.shape[1:3]:
        raise ValueError("Mask shape must match ROI dimensions")
    if radius < 2:
        raise ValueError("Spatial fit radius must be at least 2")

    frames, height, width, _ = roi_stack.shape
    selected = np.argwhere(mask > 0)
    background = np.zeros_like(roi_stack, dtype=np.float32)
    spatial_rmse = np.full((height, width), 255.0, dtype=np.float32)
    source = roi_stack.astype(np.float32)

    for py, px in selected:
        y1, y2 = max(0, py - radius), min(height, py + radius + 1)
        x1, x2 = max(0, px - radius), min(width, px + radius + 1)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        valid = mask[y1:y2, x1:x2] == 0
        dx = (xx[valid] - px).astype(np.float32)
        dy = (yy[valid] - py).astype(np.float32)
        if dx.size < 12:
            continue
        design = np.stack((np.ones_like(dx), dx, dy), axis=1)
        distance = np.sqrt(dx * dx + dy * dy)
        weights = np.exp(-0.5 * (distance / max(1.0, radius / 2.0)) ** 2)
        weighted_design = design * weights[:, None]
        normal = design.T @ weighted_design
        try:
            projection = np.linalg.solve(normal, weighted_design.T)
        except np.linalg.LinAlgError:
            projection = np.linalg.pinv(normal) @ weighted_design.T

        samples = source[:, y1:y2, x1:x2][:, valid, :]
        coefficients = np.einsum("kn,tnc->tkc", projection, samples)
        background[:, py, px, :] = coefficients[:, 0, :]
        predicted_neighbors = np.einsum("nk,tkc->tnc", design, coefficients)
        residual = samples - predicted_neighbors
        per_frame_rmse = np.sqrt(np.average(residual * residual, axis=1, weights=weights))
        spatial_rmse[py, px] = float(np.median(np.mean(per_frame_rmse, axis=1)))

    return background, spatial_rmse


def estimate_proxy_backgrounds(
    roi_stack: np.ndarray,
    mask: np.ndarray,
    radius: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose a nearby same-frame proxy whose temporal color variation best explains each logo pixel."""
    if roi_stack.ndim != 4 or roi_stack.shape[-1] != 3:
        raise ValueError("ROI stack must have shape [frames, height, width, 3]")
    if mask.shape != roi_stack.shape[1:3]:
        raise ValueError("Mask shape must match ROI dimensions")
    source = roi_stack.astype(np.float32)
    _, height, width, _ = source.shape
    background = np.zeros_like(source, dtype=np.float32)
    proxy_rmse = np.full((height, width), 255.0, dtype=np.float32)
    proxy_distance = np.zeros((height, width), dtype=np.float32)

    for py, px in np.argwhere(mask > 0):
        y1, y2 = max(0, py - radius), min(height, py + radius + 1)
        x1, x2 = max(0, px - radius), min(width, px + radius + 1)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        valid = mask[y1:y2, x1:x2] == 0
        candidates = source[:, y1:y2, x1:x2][:, valid, :]
        candidate_y = yy[valid]
        candidate_x = xx[valid]
        distance = np.sqrt((candidate_y - py) ** 2 + (candidate_x - px) ** 2).astype(np.float32)
        keep = distance >= 2.0
        candidates = candidates[:, keep, :]
        candidate_y, candidate_x, distance = candidate_y[keep], candidate_x[keep], distance[keep]
        if candidates.shape[1] == 0:
            continue

        observed = source[:, py, px, :]
        observed_centered = observed - observed.mean(axis=0, keepdims=True)
        candidate_centered = candidates - candidates.mean(axis=0, keepdims=True)
        numerator = np.sum(candidate_centered * observed_centered[:, None, :], axis=(0, 2))
        denominator = np.sum(candidate_centered * candidate_centered, axis=(0, 2)) + 1e-6
        slope = numerator / denominator
        alpha = 1.0 - slope
        intercept = np.mean(observed[:, None, :] - slope[None, :, None] * candidates, axis=0)
        watermark = intercept / np.maximum(alpha[:, None], 1e-4)
        predicted = slope[None, :, None] * candidates + intercept[None, :, :]
        rmse = np.sqrt(np.mean((observed[:, None, :] - predicted) ** 2, axis=(0, 2)))
        dynamic = np.std(
            0.114 * candidates[..., 0] + 0.587 * candidates[..., 1] + 0.299 * candidates[..., 2],
            axis=0,
        )
        neutral_spread = np.max(watermark, axis=1) - np.min(watermark, axis=1)
        plausible = (
            (alpha >= 0.02) & (alpha <= 0.92)
            & (np.mean(watermark, axis=1) >= 175.0)
            & (np.mean(watermark, axis=1) <= 285.0)
            & (neutral_spread <= 80.0)
            & (dynamic >= 12.0)
        )
        score = rmse + 0.12 * distance
        score[~plausible] = np.inf
        best = int(np.argmin(score))
        if not np.isfinite(score[best]):
            continue
        background[:, py, px, :] = candidates[:, best, :]
        proxy_rmse[py, px] = rmse[best]
        proxy_distance[py, px] = distance[best]

    return background, proxy_rmse, proxy_distance


def estimate_distribution_model(
    roi_stack: np.ndarray,
    mask: np.ndarray,
    radius: int = 10,
    confidence_threshold: float = 0.45,
    alpha_min: float = 0.03,
    alpha_max: float = 0.80,
    gamut_min: float = 0.85,
) -> tuple[AlphaModel, np.ndarray]:
    """Estimate alpha from nearby temporal distributions without borrowing donor frames."""
    source = roi_stack.astype(np.float32)
    _, height, width, _ = source.shape
    if mask.shape != (height, width):
        raise ValueError("Mask shape must match ROI dimensions")
    quantile_levels = np.linspace(5.0, 95.0, 19)
    selected_coords = np.argwhere(mask > 0)
    count = len(selected_coords)
    observed_quantiles: list[np.ndarray] = []
    chosen_quantiles: list[np.ndarray | None] = []
    candidate_watermarks = np.zeros((count, 3), dtype=np.float32)
    candidate_alpha = np.zeros(count, dtype=np.float32)
    candidate_rmse = np.full(count, 255.0, dtype=np.float32)
    candidate_distance = np.zeros(count, dtype=np.float32)

    for index, (py, px) in enumerate(selected_coords):
        y1, y2 = max(0, py - radius), min(height, py + radius + 1)
        x1, x2 = max(0, px - radius), min(width, px + radius + 1)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        valid = mask[y1:y2, x1:x2] == 0
        candidates = source[:, y1:y2, x1:x2][:, valid, :]
        candidate_y = yy[valid]
        candidate_x = xx[valid]
        distance = np.sqrt((candidate_y - py) ** 2 + (candidate_x - px) ** 2).astype(np.float32)
        keep = distance >= 2.0
        candidates, distance = candidates[:, keep, :], distance[keep]
        observed_q = np.percentile(source[:, py, px, :], quantile_levels, axis=0).astype(np.float32)
        observed_quantiles.append(observed_q)
        if candidates.shape[1] == 0:
            chosen_quantiles.append(None)
            continue
        candidate_q = np.percentile(candidates, quantile_levels, axis=0).astype(np.float32)
        observed_centered = observed_q - observed_q.mean(axis=0, keepdims=True)
        candidate_centered = candidate_q - candidate_q.mean(axis=0, keepdims=True)
        numerator = np.sum(candidate_centered * observed_centered[:, None, :], axis=(0, 2))
        denominator = np.sum(candidate_centered * candidate_centered, axis=(0, 2)) + 1e-6
        slope = numerator / denominator
        alpha = 1.0 - slope
        intercept = np.mean(observed_q[:, None, :] - slope[None, :, None] * candidate_q, axis=0)
        watermark = intercept / np.maximum(alpha[:, None], 1e-4)
        predicted = slope[None, :, None] * candidate_q + intercept[None, :, :]
        rmse = np.sqrt(np.mean((observed_q[:, None, :] - predicted) ** 2, axis=(0, 2)))
        dynamic = np.mean(candidate_q[-2] - candidate_q[1], axis=1)
        neutral_spread = np.max(watermark, axis=1) - np.min(watermark, axis=1)
        plausible = (
            (alpha >= 0.02) & (alpha <= 0.92)
            & (np.mean(watermark, axis=1) >= 175.0)
            & (np.mean(watermark, axis=1) <= 300.0)
            & (neutral_spread <= 90.0)
            & (dynamic >= 18.0)
        )
        score = rmse + 0.08 * distance
        score[~plausible] = np.inf
        best = int(np.argmin(score))
        if not np.isfinite(score[best]):
            chosen_quantiles.append(None)
            continue
        chosen_quantiles.append(candidate_q[:, best, :])
        candidate_watermarks[index] = watermark[best]
        candidate_alpha[index] = alpha[best]
        candidate_rmse[index] = rmse[best]
        candidate_distance[index] = distance[best]

    valid_candidates = candidate_rmse < 255.0
    if np.count_nonzero(valid_candidates) < 3:
        raise RuntimeError("Too few plausible distribution proxies to estimate watermark color")
    robust_cutoff = np.percentile(candidate_rmse[valid_candidates], 60)
    robust = valid_candidates & (candidate_rmse <= robust_cutoff)
    watermark_bgr = np.clip(np.median(candidate_watermarks[robust], axis=0), 180.0, 255.0).astype(np.float32)

    alpha_selected = np.zeros(count, dtype=np.float32)
    confidence_selected = np.zeros(count, dtype=np.float32)
    fit_rmse_selected = np.full(count, 255.0, dtype=np.float32)
    dynamic_selected = np.zeros(count, dtype=np.float32)
    gamut_selected = np.zeros(count, dtype=np.float32)
    for index, ((py, px), observed_q, background_q) in enumerate(
        zip(selected_coords, observed_quantiles, chosen_quantiles, strict=True)
    ):
        if background_q is None:
            continue
        delta = watermark_bgr[None, :] - background_q
        target = observed_q - background_q
        alpha = float(np.clip(np.sum(delta * target) / (np.sum(delta * delta) + 1e-6), 0.0, 0.95))
        predicted = (1.0 - alpha) * background_q + alpha * watermark_bgr[None, :]
        fit_rmse = float(np.sqrt(np.mean((observed_q - predicted) ** 2)))
        dynamic = float(np.mean(background_q[-2] - background_q[1]))
        inverse = (source[:, py, px, :] - alpha * watermark_bgr) / max(1.0 - alpha, 1e-4)
        gamut = float(np.mean(np.all((inverse >= -2.0) & (inverse <= 257.0), axis=1)))
        watermark_error = float(np.sqrt(np.mean((candidate_watermarks[index] - watermark_bgr) ** 2)))
        dynamic_confidence = np.clip(dynamic / 90.0, 0.0, 1.0)
        fit_confidence = np.exp(-((fit_rmse / 18.0) ** 2))
        color_confidence = np.exp(-((watermark_error / 45.0) ** 2))
        distance_confidence = np.exp(-((candidate_distance[index] / 14.0) ** 2))
        gamut_confidence = np.clip((gamut - 0.80) / 0.20, 0.0, 1.0)
        confidence = float(
            np.power(
                dynamic_confidence * fit_confidence * color_confidence
                * distance_confidence * gamut_confidence,
                0.2,
            )
        )
        alpha_selected[index] = alpha
        confidence_selected[index] = confidence
        fit_rmse_selected[index] = fit_rmse
        dynamic_selected[index] = dynamic
        gamut_selected[index] = gamut

    alpha_map = np.zeros((height, width), dtype=np.float32)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    fit_rmse_map = np.zeros((height, width), dtype=np.float32)
    proxy_rmse_map = np.zeros((height, width), dtype=np.float32)
    dynamic_map = np.zeros((height, width), dtype=np.float32)
    gamut_map = np.zeros((height, width), dtype=np.float32)
    distance_map = np.zeros((height, width), dtype=np.float32)
    for index, (py, px) in enumerate(selected_coords):
        alpha_map[py, px] = alpha_selected[index]
        confidence_map[py, px] = confidence_selected[index]
        fit_rmse_map[py, px] = fit_rmse_selected[index]
        proxy_rmse_map[py, px] = candidate_rmse[index]
        dynamic_map[py, px] = dynamic_selected[index]
        gamut_map[py, px] = gamut_selected[index]
        distance_map[py, px] = candidate_distance[index]

    selected = mask > 0
    transparent = selected & (alpha_map < alpha_min) & (confidence_map >= confidence_threshold)
    resolved = (
        selected & (alpha_map >= alpha_min) & (alpha_map <= alpha_max)
        & (confidence_map >= confidence_threshold) & (gamut_map >= gamut_min)
    )
    unresolved = selected & ~resolved & ~transparent
    return AlphaModel(
        watermark_bgr=watermark_bgr,
        alpha=alpha_map,
        confidence=confidence_map,
        resolved_mask=resolved.astype(np.uint8) * 255,
        transparent_mask=transparent.astype(np.uint8) * 255,
        unresolved_mask=unresolved.astype(np.uint8) * 255,
        fit_rmse=fit_rmse_map,
        spatial_rmse=proxy_rmse_map,
        background_dynamic_range=dynamic_map,
        gamut_fraction=gamut_map,
    ), distance_map


def _fit_alpha_for_watermark(
    observed: np.ndarray,
    background: np.ndarray,
    watermark_bgr: np.ndarray,
) -> np.ndarray:
    delta = watermark_bgr[None, None, :] - background
    target = observed - background
    denominator = np.sum(delta * delta, axis=(0, 2)) + 1e-6
    alpha = np.sum(delta * target, axis=(0, 2)) / denominator
    return np.clip(alpha, 0.0, 0.95)


def estimate_alpha_model(
    roi_stack: np.ndarray,
    background_stack: np.ndarray,
    spatial_rmse: np.ndarray,
    mask: np.ndarray,
    confidence_threshold: float = 0.45,
    alpha_min: float = 0.03,
    alpha_max: float = 0.80,
    iterations: int = 8,
) -> AlphaModel:
    selected = mask > 0
    observed = roi_stack[:, selected, :].astype(np.float32)
    background = background_stack[:, selected, :].astype(np.float32)
    if observed.size == 0:
        raise ValueError("Mask contains no pixels")

    watermark = np.array([245.0, 245.0, 245.0], dtype=np.float32)
    alpha_selected = np.zeros(observed.shape[1], dtype=np.float32)
    for _ in range(iterations):
        alpha_selected = _fit_alpha_for_watermark(observed, background, watermark)
        useful = alpha_selected > 0.02
        if not np.any(useful):
            raise RuntimeError("No non-zero alpha support found")
        alpha_u = alpha_selected[useful][None, :, None]
        target = observed[:, useful, :] - (1.0 - alpha_u) * background[:, useful, :]
        estimate = target / np.maximum(alpha_u, 1e-4)
        residual = observed[:, useful, :] - (
            (1.0 - alpha_u) * background[:, useful, :] + alpha_u * watermark[None, None, :]
        )
        residual_norm = np.sqrt(np.mean(residual * residual, axis=2))
        cutoff = np.percentile(residual_norm, 80)
        robust = residual_norm <= cutoff
        channel_values: list[float] = []
        for channel in range(3):
            values = estimate[..., channel][robust]
            channel_values.append(float(np.median(values)))
        watermark = np.clip(np.array(channel_values, dtype=np.float32), 180.0, 255.0)

    reconstructed_observed = (
        (1.0 - alpha_selected[None, :, None]) * background
        + alpha_selected[None, :, None] * watermark[None, None, :]
    )
    fit_rmse_selected = np.sqrt(np.mean((observed - reconstructed_observed) ** 2, axis=(0, 2)))
    background_luma_selected = (
        0.114 * background[..., 0] + 0.587 * background[..., 1] + 0.299 * background[..., 2]
    )
    dynamic_selected = np.percentile(background_luma_selected, 90, axis=0) - np.percentile(
        background_luma_selected, 10, axis=0
    )

    inverse = (
        observed - alpha_selected[None, :, None] * watermark[None, None, :]
    ) / np.maximum(1.0 - alpha_selected[None, :, None], 1e-4)
    gamut_selected = np.mean(np.all((inverse >= -2.0) & (inverse <= 257.0), axis=2), axis=0)

    spatial_selected = spatial_rmse[selected]
    dynamic_confidence = np.clip(dynamic_selected / 55.0, 0.0, 1.0)
    fit_confidence = np.exp(-((fit_rmse_selected / 24.0) ** 2))
    spatial_confidence = np.exp(-((spatial_selected / 32.0) ** 2))
    gamut_confidence = np.clip((gamut_selected - 0.80) / 0.20, 0.0, 1.0)
    confidence_selected = np.power(
        dynamic_confidence * fit_confidence * spatial_confidence * gamut_confidence,
        0.25,
    )

    alpha = np.zeros(mask.shape, dtype=np.float32)
    confidence = np.zeros(mask.shape, dtype=np.float32)
    fit_rmse = np.zeros(mask.shape, dtype=np.float32)
    dynamic_range = np.zeros(mask.shape, dtype=np.float32)
    gamut_fraction = np.zeros(mask.shape, dtype=np.float32)
    alpha[selected] = alpha_selected
    confidence[selected] = confidence_selected
    fit_rmse[selected] = fit_rmse_selected
    dynamic_range[selected] = dynamic_selected
    gamut_fraction[selected] = gamut_selected

    transparent = selected & (alpha < alpha_min) & (confidence >= confidence_threshold)
    resolved = (
        selected
        & (alpha >= alpha_min)
        & (alpha <= alpha_max)
        & (confidence >= confidence_threshold)
        & (gamut_fraction >= 0.98)
    )
    unresolved = selected & ~resolved & ~transparent
    return AlphaModel(
        watermark_bgr=watermark,
        alpha=alpha,
        confidence=confidence,
        resolved_mask=resolved.astype(np.uint8) * 255,
        transparent_mask=transparent.astype(np.uint8) * 255,
        unresolved_mask=unresolved.astype(np.uint8) * 255,
        fit_rmse=fit_rmse,
        spatial_rmse=spatial_rmse,
        background_dynamic_range=dynamic_range,
        gamut_fraction=gamut_fraction,
    )


def deblend_roi(
    roi: np.ndarray,
    model: AlphaModel,
) -> tuple[np.ndarray, np.ndarray]:
    output = roi.copy()
    selected = model.resolved_mask > 0
    if not np.any(selected):
        return output, np.zeros(model.alpha.shape, dtype=np.uint8)
    observed = roi.astype(np.float32)
    alpha = model.alpha[..., None]
    inverse = (observed - alpha * model.watermark_bgr) / np.maximum(1.0 - alpha, 1e-4)
    frame_valid = selected & np.all((inverse >= -2.0) & (inverse <= 257.0), axis=2)
    output[frame_valid] = np.clip(inverse[frame_valid], 0, 255).astype(np.uint8)
    return output, frame_valid.astype(np.uint8) * 255


def visualize_scalar(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[mask == 0] = 0
    return colored
