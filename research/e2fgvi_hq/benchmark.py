from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch

from research.e2fgvi_hq.device import (
    peak_memory_bytes,
    reset_peak_memory,
    resolve_device,
    synchronize,
)
from research.e2fgvi_hq.smoke_test import DEFAULT_CHECKPOINT, WORKSPACE, load_model
from veo_watermark_remover.config import DEFAULT_CANDIDATE_ROI
from veo_watermark_remover.diagnostics import make_contact_sheet, write_image
from veo_watermark_remover.experiment1 import _logo_likeness, _masked_temporal_mad
from veo_watermark_remover.experiment2 import _collect_roi_stack, _quantiles
from veo_watermark_remover.lama_cpu import context_crop_box, mask_in_context
from veo_watermark_remover.mask import build_shape_mask
from veo_watermark_remover.video_io import probe_video
from veo_watermark_remover.watermark import load_measurement, scaled_logo_bbox


DEFAULT_VIDEO = WORKSPACE / "samples" / "ft-vid-23.mp4"
DEFAULT_OUTPUT_DIR = WORKSPACE / "research" / "e2fgvi_hq" / "outputs"
DEFAULT_REPORT = WORKSPACE / "research" / "e2fgvi_hq" / "results" / "benchmark_report.json"
BASELINE_VIDEOS = {
    "telea": WORKSPACE / "outputs" / "experiment1" / "ft-vid-23_telea.mp4",
    "alpha": WORKSPACE / "outputs" / "experiment2" / "ft-vid-23_alpha_deblend_only.mp4",
    "lama": WORKSPACE / "outputs" / "ai_experiment1" / "ft-vid-23_lama_cpu.mp4",
}


def _report_path(path: Path) -> str:
    """Serialize relative CLI paths and absolute Colab paths safely."""
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(WORKSPACE))
    except ValueError:
        return str(path)


class PeakRssMonitor:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_bytes = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRssMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def _read_segment(video_path: Path, start_frame: int, frame_count: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    try:
        for _ in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) != frame_count:
        raise RuntimeError(
            f"Decoded {len(frames)}/{frame_count} frames from {video_path} at {start_frame}"
        )
    return np.stack(frames)


def _pad_for_hq(masked: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = masked.shape[-2:]
    height_pad = (60 - height % 60) % 60
    width_pad = (108 - width % 108) % 108
    padded = torch.cat([masked, torch.flip(masked, [3])], 3)[
        :, :, :, : height + height_pad, :
    ]
    padded = torch.cat([padded, torch.flip(padded, [4])], 4)[
        :, :, :, :, : width + width_pad
    ]
    return padded, (height, width)


def _reference_indices(
    neighbor_ids: list[int],
    length: int,
    reference_step: int,
) -> list[int]:
    return [index for index in range(0, length, reference_step) if index not in neighbor_ids]


def _center_overlap_weight(frame_index: int, center: int, neighbor_stride: int) -> float:
    distance = abs(frame_index - center)
    if distance > neighbor_stride:
        raise ValueError("Frame lies outside the temporal neighbor window")
    return float(neighbor_stride + 1 - distance)


def run_e2fgvi(
    model: torch.nn.Module,
    bgr_crops: np.ndarray,
    mask: np.ndarray,
    neighbor_stride: int = 5,
    reference_step: int = 10,
    aggregation: str = "legacy_average",
    progress: Callable[[dict[str, Any]], None] | None = None,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if aggregation not in {"legacy_average", "center_weighted"}:
        raise ValueError(f"Unsupported overlap aggregation: {aggregation}")
    if reference_step < 1:
        raise ValueError("Reference step must be positive")
    target = torch.device(device) if device is not None else next(model.parameters()).device
    rgb = bgr_crops[..., ::-1].copy()
    images = (
        torch.from_numpy(rgb)
        .permute(0, 3, 1, 2)
        .to(device=target, dtype=torch.float32)
        .div_(127.5)
        .sub_(1.0)
    )
    mask_tensor = torch.from_numpy((mask > 0).astype(np.float32))[None, None].to(target)
    masks = mask_tensor.repeat(1, len(bgr_crops), 1, 1, 1)
    images = images.unsqueeze(0)
    predictions: list[np.ndarray | None] = [None] * len(bgr_crops)
    prediction_weights = np.zeros(len(bgr_crops), dtype=np.float32)
    contributions = np.zeros(len(bgr_crops), dtype=np.int32)
    window_seconds: list[float] = []
    window_shapes: list[dict[str, Any]] = []

    with torch.inference_mode():
        for center in range(0, len(bgr_crops), neighbor_stride):
            neighbor_ids = list(
                range(
                    max(0, center - neighbor_stride),
                    min(len(bgr_crops), center + neighbor_stride + 1),
                )
            )
            reference_ids = _reference_indices(
                neighbor_ids, len(bgr_crops), reference_step
            )
            selected_ids = neighbor_ids + reference_ids
            selected_images = images[:, selected_ids]
            selected_masks = masks[:, selected_ids]
            masked_images = selected_images * (1.0 - selected_masks)
            padded, original_size = _pad_for_hq(masked_images)
            synchronize(target)
            started = time.perf_counter()
            output, _ = model(padded, num_local_frames=len(neighbor_ids))
            synchronize(target)
            elapsed = time.perf_counter() - started
            window_seconds.append(elapsed)
            window_record = {
                "center_local": center,
                "neighbors": neighbor_ids,
                "references": reference_ids,
                "input_shape": list(padded.shape),
                "seconds": round(elapsed, 6),
            }
            window_shapes.append(window_record)
            if progress is not None:
                progress(window_record)
            height, width = original_size
            predicted = output[: len(neighbor_ids), :, :height, :width]
            predicted = (
                predicted.add(1.0)
                .div_(2.0)
                .clamp_(0.0, 1.0)
                .permute(0, 2, 3, 1)
                .to("cpu")
                .numpy()
                * 255.0
            ).astype(np.uint8)
            predicted = predicted[..., ::-1]
            for offset, frame_index in enumerate(neighbor_ids):
                value = predicted[offset].astype(np.float32)
                if aggregation == "center_weighted":
                    weight = _center_overlap_weight(frame_index, center, neighbor_stride)
                    if predictions[frame_index] is None:
                        predictions[frame_index] = value * weight
                    else:
                        predictions[frame_index] += value * weight
                    prediction_weights[frame_index] += weight
                else:
                    if predictions[frame_index] is None:
                        predictions[frame_index] = value
                    else:
                        predictions[frame_index] = (predictions[frame_index] + value) * 0.5
                contributions[frame_index] += 1

    if any(value is None for value in predictions):
        missing = [index for index, value in enumerate(predictions) if value is None]
        raise RuntimeError(f"No E2FGVI prediction for local frames: {missing}")
    if aggregation == "center_weighted":
        normalized = [
            value / prediction_weights[index]
            for index, value in enumerate(predictions)
            if value is not None
        ]
    else:
        normalized = [value for value in predictions if value is not None]
    predicted_stack = np.stack([value.astype(np.uint8) for value in normalized])
    completed = bgr_crops.copy()
    completed[:, mask > 0] = predicted_stack[:, mask > 0]
    outside_difference = int(
        cv2.absdiff(bgr_crops.reshape(-1, *bgr_crops.shape[2:]), completed.reshape(-1, *completed.shape[2:]))[
            np.broadcast_to(mask == 0, completed.shape[:3]).reshape(-1, completed.shape[2])
        ].max()
    )
    return completed, {
        "window_seconds": [round(value, 6) for value in window_seconds],
        "window_seconds_summary": _quantiles(np.asarray(window_seconds)),
        "window_records": window_shapes,
        "contributions_per_frame": contributions.tolist(),
        "aggregation": aggregation,
        "aggregation_weights_per_frame": prediction_weights.tolist()
        if aggregation == "center_weighted"
        else None,
        "outside_mask_max_absolute_change": outside_difference,
    }


def _sequence_metrics(
    sequences: dict[str, np.ndarray],
    roi_offset: tuple[int, int, int, int],
    raw_mask: np.ndarray,
    final_mask: np.ndarray,
    start_frame: int,
) -> dict[str, Any]:
    roi_x, roi_y, roi_width, roi_height = roi_offset
    results: dict[str, Any] = {}
    transition_locals = {
        f"{before}_to_{before + 1}": before + 1 - start_frame
        for before in range(130, 134)
    }
    for name, crops in sequences.items():
        rois = crops[:, roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
        temporal = np.asarray(
            [
                _masked_temporal_mad(rois[index - 1], rois[index], final_mask)
                for index in range(1, len(rois))
            ],
            dtype=np.float64,
        )
        worst = int(np.argmax(temporal)) + 1 if temporal.size else 0
        transitions: dict[str, float | None] = {}
        luma_deltas: dict[str, float | None] = {}
        selected = final_mask > 0
        for transition_name, transition_local in transition_locals.items():
            if 0 < transition_local < len(rois):
                transitions[transition_name] = _masked_temporal_mad(
                    rois[transition_local - 1], rois[transition_local], final_mask
                )
                previous_gray = cv2.cvtColor(
                    rois[transition_local - 1], cv2.COLOR_BGR2GRAY
                ).astype(np.float32)
                current_gray = cv2.cvtColor(
                    rois[transition_local], cv2.COLOR_BGR2GRAY
                ).astype(np.float32)
                luma_deltas[transition_name] = float(
                    current_gray[selected].mean() - previous_gray[selected].mean()
                )
            else:
                transitions[transition_name] = None
                luma_deltas[transition_name] = None
        laplacian_energies = []
        for roi in rois:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
            laplacian_energies.append(float(laplacian[raw_mask > 0].mean()))
        transition_metrics: dict[str, float | None] = {}
        for transition_name in transition_locals:
            mad = transitions[transition_name]
            luma_delta = luma_deltas[transition_name]
            transition_metrics[f"transition_{transition_name}_mad"] = (
                round(float(mad), 6) if mad is not None else None
            )
            transition_metrics[f"transition_{transition_name}_mean_luma_delta"] = (
                round(float(luma_delta), 6) if luma_delta is not None else None
            )
        results[name] = {
            "mean_logo_likeness": round(
                float(np.mean([_logo_likeness(roi, raw_mask) for roi in rois])), 6
            ),
            "mean_masked_temporal_mad": round(float(temporal.mean()), 6),
            "worst_transition_to_absolute_frame": start_frame + worst,
            "worst_transition_mad": round(float(temporal[worst - 1]), 6),
            **transition_metrics,
            "mean_raw_mask_laplacian_energy": round(
                float(np.mean(laplacian_energies)), 6
            ),
        }
    return results


def _write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create diagnostic video: {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _write_diagnostics(
    output_dir: Path,
    sequences: dict[str, np.ndarray],
    mask: np.ndarray,
    start_frame: int,
    fps: float,
) -> list[int]:
    method_order = [
        name
        for name in ("original", "telea", "alpha", "lama", "e2fgvi_hq")
        if name in sequences
    ]
    count = len(next(iter(sequences.values())))
    desired = [108, 120, 128, 130, 131, 136, 144, 155]
    selected = [frame for frame in desired if start_frame <= frame < start_frame + count]
    if not selected:
        selected = sorted({start_frame, start_frame + count // 2, start_frame + count - 1})
    write_image(output_dir / "veo_mask_256.png", mask)
    sheets: list[np.ndarray] = []
    for absolute in selected:
        local = absolute - start_frame
        images = [sequences[name][local] for name in method_order]
        sheet = make_contact_sheet(
            images,
            [name.replace("_", " ").title() for name in method_order],
            columns=len(method_order),
        )
        write_image(output_dir / "comparisons" / f"frame_{absolute:06d}.png", sheet)
        sheets.append(sheet)
    write_image(
        output_dir / "comparison_contact_sheet.png",
        make_contact_sheet(
            sheets,
            [f"frame {value} | {value / fps:.3f}s" for value in selected],
            columns=1,
        ),
    )
    transition_frames: list[np.ndarray] = []
    transition_labels: list[str] = []
    for name in method_order:
        for absolute in range(130, 135):
            if start_frame <= absolute < start_frame + count:
                transition_frames.append(sequences[name][absolute - start_frame])
                transition_labels.append(f"{name} f{absolute}")
    if transition_frames:
        write_image(
            output_dir / "transitions_130_134.png",
            make_contact_sheet(transition_frames, transition_labels, columns=5),
        )
    _write_video(output_dir / "e2fgvi_hq_crop.mp4", sequences["e2fgvi_hq"], fps)
    comparison_video = np.concatenate(
        [sequences[name] for name in method_order],
        axis=2,
    )
    _write_video(output_dir / "five_method_comparison.mp4", comparison_video, fps)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="E2FGVI-HQ Veo benchmark")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--start-frame", type=int, default=108)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--crop-size", type=int, default=256, choices=[192, 224, 256])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--neighbor-stride", type=int, default=5)
    parser.add_argument("--reference-step", type=int, default=10)
    parser.add_argument(
        "--aggregation",
        choices=["legacy_average", "center_weighted"],
        default="legacy_average",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    benchmark_started = time.perf_counter()
    if args.frames < 2:
        raise ValueError("At least two frames are required for video inference")
    if args.reference_step < 1:
        raise ValueError("Reference step must be positive")
    torch.set_num_threads(max(1, args.threads))
    device_info = resolve_device(args.device)
    metadata = probe_video(args.video)
    if args.start_frame < 0 or args.start_frame + args.frames > metadata.frame_count:
        raise ValueError("Requested segment lies outside the source video")

    roi_px = DEFAULT_CANDIDATE_ROI.to_pixels(metadata.width, metadata.height)
    roi_stack = _collect_roi_stack(args.video, roi_px)
    median_roi = np.median(roi_stack, axis=0).astype(np.uint8)
    logo_bbox = scaled_logo_bbox(metadata.width, metadata.height, load_measurement())
    mask_result = build_shape_mask(median_roi, roi_px, logo_bbox, dilation=1)
    crop_box = context_crop_box(metadata.width, metadata.height, logo_bbox, args.crop_size)
    context_mask = mask_in_context(mask_result.mask, roi_px, crop_box)
    source_frames = _read_segment(args.video, args.start_frame, args.frames)
    source_crops = source_frames[
        :,
        crop_box.y : crop_box.y + crop_box.height,
        crop_box.x : crop_box.x + crop_box.width,
    ].copy()

    process = psutil.Process()
    rss_before_model = process.memory_info().rss
    model_started = time.perf_counter()
    model = load_model(args.checkpoint, device_info.device)
    synchronize(device_info.device)
    model_load_seconds = time.perf_counter() - model_started
    rss_after_model = process.memory_info().rss
    reset_peak_memory(device_info.device)
    inference_started = time.perf_counter()
    with PeakRssMonitor() as rss_monitor:
        restored, inference = run_e2fgvi(
            model,
            source_crops,
            context_mask,
            neighbor_stride=args.neighbor_stride,
            reference_step=args.reference_step,
            aggregation=args.aggregation,
            device=device_info.device,
        )
    synchronize(device_info.device)
    inference_seconds = time.perf_counter() - inference_started

    sequences = {"original": source_crops, "e2fgvi_hq": restored}
    missing_baselines: list[str] = []
    if not args.skip_baselines:
        for name, path in BASELINE_VIDEOS.items():
            if path.is_file():
                frames = _read_segment(path, args.start_frame, args.frames)
                sequences[name] = frames[
                    :,
                    crop_box.y : crop_box.y + crop_box.height,
                    crop_box.x : crop_box.x + crop_box.width,
                ].copy()
            else:
                missing_baselines.append(name)
    if not args.skip_baselines and missing_baselines:
        raise FileNotFoundError(f"Missing baseline outputs: {missing_baselines}")

    roi_offset = (
        roi_px[0] - crop_box.x,
        roi_px[1] - crop_box.y,
        roi_px[2],
        roi_px[3],
    )
    metrics = _sequence_metrics(
        sequences,
        roi_offset,
        mask_result.raw_mask,
        mask_result.mask,
        args.start_frame,
    )
    selected = _write_diagnostics(
        args.output_dir, sequences, context_mask, args.start_frame, metadata.fps
    )
    report: dict[str, Any] = {
        "experiment": f"E2FGVI-HQ {args.device.upper()} proof of concept",
        "source": _report_path(args.video),
        "segment": {
            "start_frame": args.start_frame,
            "end_frame_inclusive": args.start_frame + args.frames - 1,
            "frame_count": args.frames,
            "fps": metadata.fps,
            "duration_seconds": args.frames / metadata.fps,
            "contains_transition_130_to_131": args.start_frame <= 130
            and args.start_frame + args.frames > 131,
        },
        "runtime": {
            "device": args.device,
            "device_name": device_info.name,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "threads": torch.get_num_threads(),
            "model_load_seconds": round(model_load_seconds, 6),
            "inference_total_seconds": round(inference_seconds, 6),
            "inference_seconds_per_output_frame": round(inference_seconds / args.frames, 6),
            "inference_output_fps": round(args.frames / inference_seconds, 6),
            "total_runtime_seconds": round(time.perf_counter() - benchmark_started, 6),
            "rss_before_model_mb": round(rss_before_model / 1024**2, 3),
            "rss_after_model_mb": round(rss_after_model / 1024**2, 3),
            "peak_rss_mb": round(rss_monitor.peak_bytes / 1024**2, 3),
            "peak_vram_mb": (
                round(peak_memory_bytes(device_info.device) / 1024**2, 3)
                if peak_memory_bytes(device_info.device) is not None
                else None
            ),
            "total_vram_mb": (
                round(device_info.total_memory_bytes / 1024**2, 3)
                if device_info.total_memory_bytes is not None
                else None
            ),
        },
        "model": {
            "name": "E2FGVI-HQ-CVPR22",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint": _report_path(args.checkpoint),
            "checkpoint_sha256": "afff989d41205598a79ce24630b9c83af4b0a06f45b137979a25937d94c121a5",
            "mmcv_replacement": "isolated torchvision.ops.deform_conv2d compatibility shim",
        },
        "crop": {
            "x": crop_box.x,
            "y": crop_box.y,
            "width": crop_box.width,
            "height": crop_box.height,
            "hq_internal_padded_height": ((crop_box.height + 59) // 60) * 60,
            "hq_internal_padded_width": ((crop_box.width + 107) // 108) * 108,
        },
        "mask": {
            "raw_pixels": mask_result.raw_pixel_count,
            "final_pixels": mask_result.final_pixel_count,
            "context_pixels": int(np.count_nonzero(context_mask)),
            "dilation": 1,
            "upstream_extra_dilation_iterations": 0,
            "selection_strategy": mask_result.selection_strategy,
            "detected_component_count": mask_result.detected_component_count,
        },
        "inference_windows": inference,
        "configuration": {
            "neighbor_stride": args.neighbor_stride,
            "reference_step": args.reference_step,
            "aggregation": args.aggregation,
        },
        "cpu_baselines_seconds_per_frame": {
            "benchmark_48_frames": 2.487,
            "full_video_192_frames": 6.622,
        },
        "metrics": metrics,
        "diagnostic_frames": selected,
        "baseline_sources": {
            name: str(path.relative_to(WORKSPACE)) for name, path in BASELINE_VIDEOS.items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
