from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .alpha_recovery import deblend_roi, estimate_distribution_model, visualize_scalar
from .config import RelativeROI
from .diagnostics import make_contact_sheet, write_image
from .experiment0 import select_frame_indices
from .experiment1 import _finish_encoder, _logo_likeness, _masked_temporal_mad, _start_encoder
from .inpaint import inpaint_roi
from .mask import build_shape_mask, mask_overlay, save_mask_diagnostics
from .video_io import probe_video
from .watermark import load_measurement, scaled_logo_bbox


def _collect_roi_stack(video_path: Path, roi_px: tuple[int, int, int, int]) -> np.ndarray:
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
    return np.stack(crops)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: 0.0 for key in ("min", "p25", "median", "p75", "max")}
    result = np.percentile(values, [0, 25, 50, 75, 100])
    return dict(zip(("min", "p25", "median", "p75", "max"), map(float, result), strict=True))


def run_experiment2(
    video_path: Path,
    output_dir: Path,
    diagnostics_dir: Path,
    relative_roi: RelativeROI,
    sample_count: int = 8,
    confidence_threshold: float = 0.45,
    gamut_min: float = 0.85,
) -> dict[str, object]:
    started = time.perf_counter()
    metadata = probe_video(video_path)
    roi_px = relative_roi.to_pixels(metadata.width, metadata.height)
    measurement = load_measurement()
    logo_bbox = scaled_logo_bbox(metadata.width, metadata.height, measurement)
    roi_stack = _collect_roi_stack(video_path, roi_px)
    median_roi = np.median(roi_stack, axis=0).astype(np.uint8)
    mask_result = build_shape_mask(median_roi, roi_px, logo_bbox, dilation=1)
    model, proxy_distance = estimate_distribution_model(
        roi_stack,
        mask_result.mask,
        confidence_threshold=confidence_threshold,
        gamut_min=gamut_min,
    )

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mask_diagnostics(diagnostics_dir, mask_result)
    write_image(diagnostics_dir / "alpha_mask.png", visualize_scalar(model.alpha, mask_result.mask))
    write_image(diagnostics_dir / "confidence_map.png", visualize_scalar(model.confidence, mask_result.mask))
    write_image(diagnostics_dir / "resolved_mask.png", model.resolved_mask)
    write_image(diagnostics_dir / "transparent_mask.png", model.transparent_mask)
    write_image(diagnostics_dir / "unresolved_mask.png", model.unresolved_mask)
    write_image(diagnostics_dir / "resolved_overlay.png", mask_overlay(median_roi, model.resolved_mask))
    color_swatch = np.full((80, 240, 3), model.watermark_bgr, dtype=np.uint8)
    write_image(diagnostics_dir / "watermark_color_bgr.png", color_swatch)

    output_path = output_dir / f"{video_path.stem}_alpha_deblend_only.mp4"
    encoder = _start_encoder(video_path, output_path, metadata)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        encoder.kill()
        raise RuntimeError(f"OpenCV could not open {video_path}")

    representative = set(select_frame_indices(metadata.frame_count, sample_count))
    x, y, width, height = roi_px
    comparisons: list[np.ndarray] = []
    comparison_labels: list[str] = []
    likeness = {"original": [], "telea": [], "alpha_deblend": []}
    temporal_mad = {"original": [], "telea": [], "alpha_deblend": []}
    previous: dict[str, np.ndarray | None] = {
        "original": None, "telea": None, "alpha_deblend": None,
    }
    applied_counts: list[int] = []
    applied_raw_counts: list[int] = []
    changed_magnitude: list[float] = []
    worst_transition = {
        "original": (-1.0, -1), "telea": (-1.0, -1), "alpha_deblend": (-1.0, -1),
    }
    frame_index = 0
    processing_seconds = {"telea": 0.0, "alpha_deblend": 0.0}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            original_roi = frame[y:y + height, x:x + width].copy()
            method_started = time.perf_counter()
            telea_roi = inpaint_roi(original_roi, mask_result.mask, "telea", 3)
            processing_seconds["telea"] += time.perf_counter() - method_started
            method_started = time.perf_counter()
            deblended_roi, applied_mask = deblend_roi(original_roi, model)
            processing_seconds["alpha_deblend"] += time.perf_counter() - method_started

            current = {
                "original": original_roi,
                "telea": telea_roi,
                "alpha_deblend": deblended_roi,
            }
            for name, roi in current.items():
                old = previous[name]
                if old is not None:
                    mad = _masked_temporal_mad(old, roi, mask_result.mask)
                    temporal_mad[name].append(mad)
                    if mad > worst_transition[name][0]:
                        worst_transition[name] = (mad, frame_index)
                previous[name] = roi
                likeness[name].append(_logo_likeness(roi, mask_result.raw_mask))

            applied = applied_mask > 0
            applied_counts.append(int(np.count_nonzero(applied)))
            applied_raw_counts.append(int(np.count_nonzero(applied & (mask_result.raw_mask > 0))))
            if np.any(applied):
                changed_magnitude.append(float(cv2.absdiff(original_roi, deblended_roi)[applied].mean()))

            output_frame = frame.copy()
            output_frame[y:y + height, x:x + width] = deblended_roi
            if encoder.stdin is None:
                raise RuntimeError("FFmpeg alpha-deblend encoder stdin is unavailable")
            encoder.stdin.write(output_frame.tobytes())

            if frame_index in representative:
                applied_overlay = mask_overlay(original_roi, applied_mask)
                comparison = make_contact_sheet(
                    [original_roi, telea_roi, deblended_roi, applied_overlay],
                    ["Original", "TELEA baseline", "Alpha deblend only", "Applied pixels"],
                    columns=4,
                )
                write_image(diagnostics_dir / "comparisons" / f"frame_{frame_index:06d}.png", comparison)
                comparisons.append(comparison)
                comparison_labels.append(f"frame {frame_index} | {frame_index / metadata.fps:.3f}s")
            frame_index += 1
    except Exception:
        encoder.kill()
        raise
    finally:
        capture.release()
    _finish_encoder(encoder, "alpha-deblend")

    write_image(
        diagnostics_dir / "comparison_contact_sheet.png",
        make_contact_sheet(comparisons, comparison_labels, columns=1),
    )
    output_metadata = probe_video(output_path)
    selected = mask_result.mask > 0
    raw = mask_result.raw_mask > 0
    resolved = model.resolved_mask > 0
    unresolved = model.unresolved_mask > 0
    no_proxy = selected & (model.fit_rmse >= 254.0)
    low_confidence = selected & ~no_proxy & (model.confidence < confidence_threshold)
    alpha_too_low = selected & ~no_proxy & ~low_confidence & (model.alpha < 0.03)
    alpha_too_high = selected & ~no_proxy & ~low_confidence & (model.alpha > 0.80)
    gamut_failed = unresolved & ~no_proxy & ~low_confidence & ~alpha_too_low & ~alpha_too_high

    report: dict[str, object] = {
        "experiment": 2,
        "cpu_only": True,
        "prohibited_methods_used": {"inpaint_for_output": False, "optical_flow": False, "temporal_reconstruction": False},
        "input": asdict(metadata),
        "roi_pixels": dict(zip(("x", "y", "width", "height"), roi_px, strict=True)),
        "model": {
            "equation": "I = B * (1 - alpha) + W * alpha",
            "estimator": "nearby temporal-distribution quantile matching",
            "watermark_bgr": [round(float(value), 6) for value in model.watermark_bgr],
            "confidence_threshold": confidence_threshold,
            "gamut_min": gamut_min,
            "alpha_safe_range": [0.03, 0.80],
            "alpha_quantiles_all_mask": _quantiles(model.alpha[selected]),
            "alpha_quantiles_resolved": _quantiles(model.alpha[resolved]),
            "confidence_quantiles": _quantiles(model.confidence[selected]),
            "proxy_distance_quantiles": _quantiles(proxy_distance[selected]),
        },
        "coverage": {
            "mask_pixels": int(np.count_nonzero(selected)),
            "raw_logo_pixels": int(np.count_nonzero(raw)),
            "resolved_pixels": int(np.count_nonzero(resolved)),
            "resolved_fraction_of_mask": round(float(np.mean(resolved[selected])), 6),
            "resolved_raw_logo_pixels": int(np.count_nonzero(resolved & raw)),
            "resolved_fraction_of_raw_logo": round(float(np.mean(resolved[raw])), 6),
            "transparent_pixels": int(np.count_nonzero(model.transparent_mask)),
            "unresolved_pixels": int(np.count_nonzero(unresolved)),
            "unresolved_reasons": {
                "no_plausible_distribution_proxy": int(np.count_nonzero(no_proxy)),
                "low_confidence": int(np.count_nonzero(low_confidence)),
                "alpha_below_safe_range": int(np.count_nonzero(alpha_too_low)),
                "alpha_above_safe_range": int(np.count_nonzero(alpha_too_high)),
                "gamut_or_other_gate": int(np.count_nonzero(gamut_failed)),
            },
            "per_frame_applied_pixels": _quantiles(np.array(applied_counts, dtype=np.float32)),
            "per_frame_applied_raw_logo_pixels": _quantiles(
                np.array(applied_raw_counts, dtype=np.float32)
            ),
        },
        "metrics": {
            "mean_logo_likeness": {key: round(float(np.mean(value)), 6) for key, value in likeness.items()},
            "mean_masked_temporal_mad": {key: round(float(np.mean(value)), 6) for key, value in temporal_mad.items()},
            "worst_masked_transition": {
                key: {"to_frame": value[1], "mad": round(value[0], 6)}
                for key, value in worst_transition.items()
            },
            "mean_absolute_change_on_applied_pixels": round(float(np.mean(changed_magnitude)), 6),
        },
        "processing_seconds": {
            "telea_comparison_only": round(processing_seconds["telea"], 6),
            "alpha_deblend_only": round(processing_seconds["alpha_deblend"], 6),
            "total_including_model_and_encoding": round(time.perf_counter() - started, 6),
        },
        "output": {"path": str(output_path), "metadata": asdict(output_metadata)},
        "representative_frames": sorted(representative),
        "warning": "Diagnostic alpha-deblend-only output; unresolved pixels remain identical to input.",
    }
    (diagnostics_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
