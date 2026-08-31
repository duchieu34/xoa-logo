from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict
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
from research.e2fgvi_hq.benchmark import (
    BASELINE_VIDEOS,
    DEFAULT_VIDEO,
    PeakRssMonitor,
    _quantiles,
    run_e2fgvi,
)
from research.e2fgvi_hq.smoke_test import DEFAULT_CHECKPOINT, WORKSPACE, load_model
from veo_watermark_remover.config import DEFAULT_CANDIDATE_ROI
from veo_watermark_remover.diagnostics import make_contact_sheet, write_image
from veo_watermark_remover.experiment1 import (
    _finish_encoder,
    _logo_likeness,
    _masked_temporal_mad,
    _start_encoder,
)
from veo_watermark_remover.experiment2 import _collect_roi_stack
from veo_watermark_remover.lama_cpu import context_crop_box, mask_in_context
from veo_watermark_remover.mask import build_shape_mask
from veo_watermark_remover.video_io import probe_video
from veo_watermark_remover.watermark import load_measurement, scaled_logo_bbox


DEFAULT_OUTPUT_DIR = WORKSPACE / "research" / "e2fgvi_hq" / "outputs" / "full_validation"
DEFAULT_REPORT = (
    WORKSPACE / "research" / "e2fgvi_hq" / "results" / "full_validation_report.json"
)


def _report_path(path: Path) -> str:
    """Prefer a workspace-relative path, but support Colab /content paths."""
    try:
        return str(path.relative_to(WORKSPACE))
    except ValueError:
        return str(path)


def _read_crop_video(video_path: Path, crop: Any, frame_count: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    crops: list[np.ndarray] = []
    try:
        for _ in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            crops.append(
                frame[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width].copy()
            )
    finally:
        capture.release()
    if len(crops) != frame_count:
        raise RuntimeError(f"Decoded {len(crops)}/{frame_count} frames from {video_path}")
    return np.stack(crops)


def _audio_sha256(video_path: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            "-f",
            "adts",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _encode_full_video(
    input_path: Path,
    output_path: Path,
    restored_crops: np.ndarray,
    crop: Any,
) -> float:
    metadata = probe_video(input_path)
    encoder = _start_encoder(input_path, output_path, metadata)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        encoder.kill()
        raise RuntimeError(f"OpenCV could not open {input_path}")
    started = time.perf_counter()
    frame_index = 0
    try:
        while frame_index < len(restored_crops):
            ok, frame = capture.read()
            if not ok:
                break
            frame[
                crop.y : crop.y + crop.height,
                crop.x : crop.x + crop.width,
            ] = restored_crops[frame_index]
            if encoder.stdin is None:
                raise RuntimeError("FFmpeg full-validation encoder stdin is unavailable")
            encoder.stdin.write(frame.tobytes())
            frame_index += 1
    except Exception:
        encoder.kill()
        raise
    finally:
        capture.release()
    if frame_index != len(restored_crops):
        encoder.kill()
        raise RuntimeError(f"Encoded {frame_index}/{len(restored_crops)} frames")
    _finish_encoder(encoder, "e2fgvi-hq-full-validation")
    return time.perf_counter() - started


def _gray_mask_values(crops: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = []
    selected = mask > 0
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        values.append(float(gray[selected].mean()))
    return np.asarray(values, dtype=np.float64)


def _sequence_metrics(crops: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    temporal = np.asarray(
        [
            _masked_temporal_mad(crops[index - 1], crops[index], mask)
            for index in range(1, len(crops))
        ],
        dtype=np.float64,
    )
    luma = _gray_mask_values(crops, mask)
    luma_delta = np.diff(luma)
    order = np.argsort(temporal)[::-1]
    return {
        "mean_temporal_mad": round(float(temporal.mean()), 6),
        "temporal_mad_quantiles": {
            key: round(value, 6) for key, value in _quantiles(temporal).items()
        },
        "worst_transitions": [
            {
                "from_frame": int(index),
                "to_frame": int(index + 1),
                "mad": round(float(temporal[index]), 6),
                "mean_luma_delta": round(float(luma_delta[index]), 6),
            }
            for index in order[:10]
        ],
        "all_transition_mad": [round(float(value), 6) for value in temporal],
        "all_mean_luma_delta": [round(float(value), 6) for value in luma_delta],
    }


def _dark_patch_proxy(crops: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = mask > 0
    dilated = cv2.dilate(mask, np.ones((11, 11), dtype=np.uint8), iterations=1)
    ring = (dilated > 0) & ~selected
    differences = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        differences.append(float(gray[selected].mean() - gray[ring].mean()))
    values = np.asarray(differences)
    darkest = np.argsort(values)[:10]
    return {
        "definition": "mean(mask luma) - mean(11x11 dilated ring luma); negative is darker",
        "mean": round(float(values.mean()), 6),
        "minimum": round(float(values.min()), 6),
        "darkest_frames": [
            {"frame": int(index), "difference": round(float(values[index]), 6)}
            for index in darkest
        ],
    }


def _crop_boundary_max_difference(
    original: np.ndarray, restored: np.ndarray, border_width: int = 2
) -> int:
    border = np.zeros(original.shape[1:3], dtype=bool)
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    difference = cv2.absdiff(original, restored)
    return int(difference[:, border].max())


def _write_transition_diagnostics(
    output_dir: Path,
    sequences: dict[str, np.ndarray],
    transitions: list[dict[str, Any]],
    mask: np.ndarray,
) -> None:
    names = tuple(
        name
        for name in ("original", "telea", "lama", "e2fgvi_hq")
        if name in sequences
    )
    for rank, transition in enumerate(transitions, start=1):
        before = int(transition["from_frame"])
        after = int(transition["to_frame"])
        images: list[np.ndarray] = []
        labels: list[str] = []
        for name in names:
            mad = _masked_temporal_mad(sequences[name][before], sequences[name][after], mask)
            images.extend([sequences[name][before], sequences[name][after]])
            labels.extend([f"{name} f{before}", f"{name} f{after} | MAD {mad:.2f}"])
        write_image(
            output_dir
            / "top_transitions"
            / f"top_{rank:02d}_frame_{before:06d}_to_{after:06d}.png",
            make_contact_sheet(images, labels, columns=4),
        )


def _window_progress(record: dict[str, Any]) -> None:
    print(
        "window "
        f"center={record['center_local']:03d} "
        f"frames={record['input_shape'][1]:02d} "
        f"seconds={record['seconds']:.3f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-video E2FGVI-HQ validation")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--crop-size", type=int, default=192, choices=[192])
    parser.add_argument("--neighbor-stride", type=int, default=5, choices=[5])
    parser.add_argument("--reference-step", type=int, default=10, choices=[10])
    parser.add_argument("--aggregation", default="legacy_average", choices=["legacy_average"])
    parser.add_argument("--threads", type=int, default=4, choices=[4])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    validation_started = time.perf_counter()
    torch.set_num_threads(args.threads)
    device_info = resolve_device(args.device)
    metadata = probe_video(args.video)
    roi_px = DEFAULT_CANDIDATE_ROI.to_pixels(metadata.width, metadata.height)
    roi_stack = _collect_roi_stack(args.video, roi_px)
    median_roi = np.median(roi_stack, axis=0).astype(np.uint8)
    logo_bbox = scaled_logo_bbox(metadata.width, metadata.height, load_measurement())
    mask_result = build_shape_mask(median_roi, roi_px, logo_bbox, dilation=1)
    crop = context_crop_box(metadata.width, metadata.height, logo_bbox, args.crop_size)
    context_mask = mask_in_context(mask_result.mask, roi_px, crop)
    original_crops = _read_crop_video(args.video, crop, metadata.frame_count)

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
        restored_crops, inference = run_e2fgvi(
            model,
            original_crops,
            context_mask,
            neighbor_stride=args.neighbor_stride,
            reference_step=args.reference_step,
            aggregation=args.aggregation,
            progress=_window_progress,
            device=device_info.device,
        )
    synchronize(device_info.device)
    inference_seconds = time.perf_counter() - inference_started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_video = args.output_dir / f"{args.video.stem}_e2fgvi_hq_{args.device}_full.mp4"
    encode_seconds = _encode_full_video(args.video, output_video, restored_crops, crop)
    output_metadata = probe_video(output_video)
    input_audio_hash = _audio_sha256(args.video)
    output_audio_hash = _audio_sha256(output_video)

    sequences = {"original": original_crops, "e2fgvi_hq": restored_crops}
    for name in ("telea", "lama"):
        path = BASELINE_VIDEOS[name]
        if path.is_file():
            sequences[name] = _read_crop_video(path, crop, metadata.frame_count)

    sequence_metrics = {
        name: _sequence_metrics(crops, context_mask) for name, crops in sequences.items()
    }
    top_transitions = sequence_metrics["e2fgvi_hq"]["worst_transitions"]
    _write_transition_diagnostics(args.output_dir, sequences, top_transitions, context_mask)
    local_roi = (
        roi_px[0] - crop.x,
        roi_px[1] - crop.y,
        roi_px[2],
        roi_px[3],
    )
    local_x, local_y, roi_width, roi_height = local_roi
    logo_likeness = {}
    for name, crops in sequences.items():
        rois = crops[:, local_y : local_y + roi_height, local_x : local_x + roi_width]
        logo_likeness[name] = round(
            float(np.mean([_logo_likeness(roi, mask_result.raw_mask) for roi in rois])),
            6,
        )

    report: dict[str, Any] = {
        "experiment": f"E2FGVI-HQ full-video {args.device.upper()} validation",
        "cpu_only": args.device == "cpu",
        "configuration": {
            "crop_size": args.crop_size,
            "internal_padding": [240, 216],
            "neighbor_stride": args.neighbor_stride,
            "reference_step": args.reference_step,
            "aggregation": args.aggregation,
            "threads": args.threads,
        },
        "input": asdict(metadata),
        "runtime": {
            "device": args.device,
            "device_name": device_info.name,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "model_load_seconds": round(model_load_seconds, 6),
            "inference_total_seconds": round(inference_seconds, 6),
            "inference_seconds_per_frame": round(inference_seconds / metadata.frame_count, 6),
            "inference_throughput_fps": round(metadata.frame_count / inference_seconds, 6),
            "encode_mux_seconds": round(encode_seconds, 6),
            "total_validation_seconds": round(time.perf_counter() - validation_started, 6),
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
        "mask": {
            "raw_pixels": mask_result.raw_pixel_count,
            "final_pixels": mask_result.final_pixel_count,
            "outside_mask_max_absolute_change_before_encoding": inference[
                "outside_mask_max_absolute_change"
            ],
            "crop_boundary_max_absolute_change_before_encoding": _crop_boundary_max_difference(
                original_crops, restored_crops
            ),
        },
        "inference_windows": inference,
        "quality": {
            "sequence_metrics": sequence_metrics,
            "mean_logo_likeness": logo_likeness,
            "e2fgvi_dark_patch_proxy": _dark_patch_proxy(restored_crops, context_mask),
            "top_10_e2fgvi_transitions": top_transitions,
        },
        "output": {
            "path": _report_path(output_video),
            "metadata": asdict(output_metadata),
            "input_audio_adts_sha256": input_audio_hash,
            "output_audio_adts_sha256": output_audio_hash,
            "audio_bitstream_match": input_audio_hash == output_audio_hash,
        },
        "scope_guards": {
            "upstream_modified": False,
            "main_pipeline_integrated": False,
            "new_smoothing_or_method": False,
            "full_frame_model_inference": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "runtime": report["runtime"], "output": report["output"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
