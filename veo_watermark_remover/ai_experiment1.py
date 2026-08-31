from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .alpha_recovery import deblend_roi, estimate_distribution_model
from .config import RelativeROI
from .diagnostics import make_contact_sheet, write_image
from .experiment0 import select_frame_indices
from .experiment1 import _finish_encoder, _logo_likeness, _masked_temporal_mad, _start_encoder
from .experiment2 import _collect_roi_stack, _quantiles
from .inpaint import inpaint_roi
from .lama_cpu import LamaCpu, context_crop_box, mask_in_context
from .mask import build_shape_mask, mask_overlay
from .video_io import probe_video
from .watermark import load_measurement, scaled_logo_bbox


CONTEXT_SIZES = (192, 256)


def _read_selected_frames(video_path: Path, selected: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    frames: dict[int, np.ndarray] = {}
    index = 0
    try:
        while selected - frames.keys():
            ok, frame = capture.read()
            if not ok:
                break
            if index in selected:
                frames[index] = frame.copy()
            index += 1
    finally:
        capture.release()
    missing = selected - frames.keys()
    if missing:
        raise RuntimeError(f"Could not decode selected frames: {sorted(missing)}")
    return frames


def _masked_laplacian_energy(roi: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    selected = mask > 0
    return float(np.mean(laplacian[selected])) if np.any(selected) else 0.0


def _sequence_metrics(
    sequences: dict[str, np.ndarray], raw_mask: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    likeness: dict[str, float] = {}
    temporal_mad: dict[str, float] = {}
    worst: dict[str, dict[str, float | int]] = {}
    transition_130_131: dict[str, float] = {}
    laplacian: dict[str, float] = {}
    for name, sequence in sequences.items():
        likeness[name] = float(np.mean([_logo_likeness(roi, raw_mask) for roi in sequence]))
        differences = [
            _masked_temporal_mad(sequence[index - 1], sequence[index], mask)
            for index in range(1, len(sequence))
        ]
        worst_index = int(np.argmax(differences)) + 1
        temporal_mad[name] = float(np.mean(differences))
        worst[name] = {"to_frame": worst_index, "mad": float(differences[worst_index - 1])}
        transition_130_131[name] = (
            _masked_temporal_mad(sequence[130], sequence[131], mask)
            if len(sequence) > 131 else 0.0
        )
        laplacian[name] = float(np.mean([
            _masked_laplacian_energy(roi, raw_mask) for roi in sequence
        ]))
    return {
        "mean_logo_likeness": {key: round(value, 6) for key, value in likeness.items()},
        "mean_masked_temporal_mad": {
            key: round(value, 6) for key, value in temporal_mad.items()
        },
        "worst_masked_transition": {
            key: {"to_frame": value["to_frame"], "mad": round(float(value["mad"]), 6)}
            for key, value in worst.items()
        },
        "transition_130_to_131_mad": {
            key: round(value, 6) for key, value in transition_130_131.items()
        },
        "mean_raw_mask_laplacian_energy": {
            key: round(value, 6) for key, value in laplacian.items()
        },
    }


def run_ai_experiment1(
    video_path: Path,
    output_dir: Path,
    diagnostics_dir: Path,
    relative_roi: RelativeROI,
    model_path: Path,
    sample_count: int = 8,
    selected_context_size: int = 256,
) -> dict[str, object]:
    if selected_context_size not in CONTEXT_SIZES:
        raise ValueError(f"Selected LaMa context must be one of {CONTEXT_SIZES}")
    started = time.perf_counter()
    metadata = probe_video(video_path)
    roi_px = relative_roi.to_pixels(metadata.width, metadata.height)
    roi_x, roi_y, roi_width, roi_height = roi_px
    roi_stack = _collect_roi_stack(video_path, roi_px)
    median_roi = np.median(roi_stack, axis=0).astype(np.uint8)
    logo_bbox = scaled_logo_bbox(metadata.width, metadata.height, load_measurement())
    mask_result = build_shape_mask(median_roi, roi_px, logo_bbox, dilation=1)
    alpha_model, _ = estimate_distribution_model(roi_stack, mask_result.mask)
    lama = LamaCpu(model_path)

    context_boxes = {
        size: context_crop_box(metadata.width, metadata.height, logo_bbox, size)
        for size in CONTEXT_SIZES
    }
    context_masks = {
        size: mask_in_context(mask_result.mask, roi_px, context_boxes[size])
        for size in CONTEXT_SIZES
    }
    representative = set(select_frame_indices(metadata.frame_count, sample_count))
    difficult = {index for index in (128, 129, 130, 131) if index < metadata.frame_count}
    diagnostic_frames = sorted(representative | difficult)
    selected_frames = _read_selected_frames(video_path, set(diagnostic_frames))

    # Warm up TorchScript before measuring either context size.
    warm_frame = selected_frames[diagnostic_frames[0]]
    warm_box = context_boxes[selected_context_size]
    lama.predict(
        warm_frame[warm_box.y:warm_box.y + warm_box.height, warm_box.x:warm_box.x + warm_box.width],
        context_masks[selected_context_size],
    )

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_results: dict[int, dict[int, np.ndarray]] = {size: {} for size in CONTEXT_SIZES}
    candidate_timings: dict[int, list[float]] = {size: [] for size in CONTEXT_SIZES}
    candidate_likeness: dict[int, list[float]] = {size: [] for size in CONTEXT_SIZES}
    candidate_laplacian: dict[int, list[float]] = {size: [] for size in CONTEXT_SIZES}
    outside_mask_max_difference = 0
    for size in CONTEXT_SIZES:
        box = context_boxes[size]
        context_mask = context_masks[size]
        write_image(diagnostics_dir / f"context_{size}_mask.png", context_mask)
        for frame_index in diagnostic_frames:
            frame = selected_frames[frame_index]
            crop = frame[box.y:box.y + box.height, box.x:box.x + box.width].copy()
            inference_started = time.perf_counter()
            restored_crop = lama.predict(crop, context_mask)
            candidate_timings[size].append(time.perf_counter() - inference_started)
            outside = context_mask == 0
            outside_mask_max_difference = max(
                outside_mask_max_difference,
                int(cv2.absdiff(crop, restored_crop)[outside].max()),
            )
            restored_roi = restored_crop[
                roi_y - box.y:roi_y - box.y + roi_height,
                roi_x - box.x:roi_x - box.x + roi_width,
            ].copy()
            candidate_results[size][frame_index] = restored_roi
            candidate_likeness[size].append(_logo_likeness(restored_roi, mask_result.raw_mask))
            candidate_laplacian[size].append(
                _masked_laplacian_energy(restored_roi, mask_result.raw_mask)
            )

    alpha_diagnostics: dict[int, np.ndarray] = {}
    comparisons: list[np.ndarray] = []
    comparison_labels: list[str] = []
    for frame_index in diagnostic_frames:
        original_roi = roi_stack[frame_index]
        telea_roi = inpaint_roi(original_roi, mask_result.mask, "telea", 3)
        alpha_roi, _ = deblend_roi(original_roi, alpha_model)
        alpha_diagnostics[frame_index] = alpha_roi
        comparison = make_contact_sheet(
            [
                original_roi,
                telea_roi,
                alpha_roi,
                candidate_results[192][frame_index],
                candidate_results[256][frame_index],
            ],
            ["Original", "TELEA", "Alpha-only", "LaMa 192", "LaMa 256"],
            columns=5,
        )
        write_image(
            diagnostics_dir / "comparisons" / f"frame_{frame_index:06d}.png", comparison
        )
        if frame_index in representative:
            comparisons.append(comparison)
            comparison_labels.append(f"frame {frame_index} | {frame_index / metadata.fps:.3f}s")
    write_image(
        diagnostics_dir / "comparison_contact_sheet.png",
        make_contact_sheet(comparisons, comparison_labels, columns=1),
    )
    transition_sheet = make_contact_sheet(
        [
            candidate_results[256][130], candidate_results[256][131],
            candidate_results[192][130], candidate_results[192][131],
        ],
        ["LaMa256 f130", "LaMa256 f131", "LaMa192 f130", "LaMa192 f131"],
        columns=2,
    )
    write_image(diagnostics_dir / "transition_130_131.png", transition_sheet)

    selected_box = context_boxes[selected_context_size]
    selected_mask = context_masks[selected_context_size]
    output_path = output_dir / f"{video_path.stem}_lama_cpu.mp4"
    encoder = _start_encoder(video_path, output_path, metadata)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        encoder.kill()
        raise RuntimeError(f"OpenCV could not open {video_path}")
    sequences: dict[str, list[np.ndarray]] = {
        "original": [], "telea": [], "alpha": [], "lama": [],
    }
    timings: dict[str, list[float]] = {"telea": [], "alpha": [], "lama": []}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            original_roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width].copy()
            method_started = time.perf_counter()
            telea_roi = inpaint_roi(original_roi, mask_result.mask, "telea", 3)
            timings["telea"].append(time.perf_counter() - method_started)
            method_started = time.perf_counter()
            alpha_roi, _ = deblend_roi(original_roi, alpha_model)
            timings["alpha"].append(time.perf_counter() - method_started)
            crop = frame[
                selected_box.y:selected_box.y + selected_box.height,
                selected_box.x:selected_box.x + selected_box.width,
            ].copy()
            method_started = time.perf_counter()
            restored_crop = lama.predict(crop, selected_mask)
            timings["lama"].append(time.perf_counter() - method_started)
            outside_mask_max_difference = max(
                outside_mask_max_difference,
                int(cv2.absdiff(crop, restored_crop)[selected_mask == 0].max()),
            )
            restored_roi = restored_crop[
                roi_y - selected_box.y:roi_y - selected_box.y + roi_height,
                roi_x - selected_box.x:roi_x - selected_box.x + roi_width,
            ].copy()
            sequences["original"].append(original_roi)
            sequences["telea"].append(telea_roi)
            sequences["alpha"].append(alpha_roi)
            sequences["lama"].append(restored_roi)
            output_frame = frame.copy()
            output_frame[
                selected_box.y:selected_box.y + selected_box.height,
                selected_box.x:selected_box.x + selected_box.width,
            ] = restored_crop
            if encoder.stdin is None:
                raise RuntimeError("FFmpeg LaMa encoder stdin is unavailable")
            encoder.stdin.write(output_frame.tobytes())
            frame_index += 1
    except Exception:
        encoder.kill()
        raise
    finally:
        capture.release()
    _finish_encoder(encoder, "lama-cpu")
    sequence_arrays = {name: np.stack(values) for name, values in sequences.items()}
    output_metadata = probe_video(output_path)
    report: dict[str, object] = {
        "experiment": "AI Experiment 1",
        "cpu_only": True,
        "prohibited_methods_used": {
            "gpu_or_cuda": False,
            "optical_flow": False,
            "video_model": False,
            "full_frame_ai_inference": False,
            "blur_or_flat_patch": False,
        },
        "input": asdict(metadata),
        "roi_pixels": dict(zip(("x", "y", "width", "height"), roi_px, strict=True)),
        "model": {
            "name": "IOPaint big-LaMa TorchScript",
            "path": str(model_path),
            "md5": lama.model_md5,
            "source": "https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py",
            "runtime": lama.runtime_info,
        },
        "mask": {
            "pixels": int(np.count_nonzero(mask_result.mask)),
            "raw_logo_pixels": int(np.count_nonzero(mask_result.raw_mask)),
            "outside_mask_max_absolute_change_before_encoding": outside_mask_max_difference,
        },
        "context_screening": {
            str(size): {
                "crop": asdict(context_boxes[size]),
                "frames": diagnostic_frames,
                "seconds_per_frame": _quantiles(np.array(candidate_timings[size], dtype=np.float32)),
                "mean_logo_likeness": round(float(np.mean(candidate_likeness[size])), 6),
                "mean_raw_mask_laplacian_energy": round(
                    float(np.mean(candidate_laplacian[size])), 6
                ),
            }
            for size in CONTEXT_SIZES
        },
        "selected_context_size": selected_context_size,
        "metrics": _sequence_metrics(sequence_arrays, mask_result.raw_mask, mask_result.mask),
        "processing_seconds": {
            name: {
                "total": round(float(np.sum(values)), 6),
                "mean_per_frame": round(float(np.mean(values)), 6),
                "fps": round(float(1.0 / np.mean(values)), 6),
            }
            for name, values in timings.items()
        },
        "total_pipeline_seconds": round(time.perf_counter() - started, 6),
        "output": {"path": str(output_path), "metadata": asdict(output_metadata)},
        "representative_frames": sorted(representative),
        "difficult_frames": sorted(difficult),
        "warning": "Per-frame LaMa diagnostic; no temporal model or motion compensation.",
    }
    (diagnostics_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_image(
        diagnostics_dir / "mask_overlay.png",
        mask_overlay(median_roi, mask_result.mask),
    )
    return report
