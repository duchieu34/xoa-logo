from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import RelativeROI
from .diagnostics import make_contact_sheet, write_image
from .experiment0 import select_frame_indices
from .inpaint import inpaint_roi
from .mask import build_shape_mask, collect_temporal_median, mask_overlay, save_mask_diagnostics
from .video_io import VideoMetadata, probe_video
from .watermark import load_measurement, scaled_logo_bbox


def _start_encoder(
    input_path: Path,
    output_path: Path,
    metadata: VideoMetadata,
) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", f"{metadata.width}x{metadata.height}",
        "-framerate", f"{metadata.fps:.12g}", "-i", "-",
        "-i", str(input_path),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-map_metadata", "1", "-shortest", "-movflags", "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _finish_encoder(process: subprocess.Popen[bytes], name: str) -> None:
    if process.stdin is not None:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg {name} encoder failed ({return_code}): {stderr.strip()}")


def _masked_temporal_mad(previous: np.ndarray, current: np.ndarray, mask: np.ndarray) -> float:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(previous_gray, current_gray)
    return float(difference[mask > 0].mean())


def _logo_likeness(roi: np.ndarray, raw_mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    score = (hsv[..., 2] / 255.0) * (1.0 - hsv[..., 1] / 255.0)
    return float(score[raw_mask > 0].mean())


def run_experiment1(
    video_path: Path,
    output_dir: Path,
    diagnostics_dir: Path,
    relative_roi: RelativeROI,
    sample_count: int = 8,
    dilation: int = 1,
    inpaint_radius: float = 3.0,
) -> dict[str, object]:
    started = time.perf_counter()
    metadata = probe_video(video_path)
    roi_px = relative_roi.to_pixels(metadata.width, metadata.height)
    measurement = load_measurement()
    logo_bbox = scaled_logo_bbox(metadata.width, metadata.height, measurement)
    median_roi, frames_used_for_mask = collect_temporal_median(video_path, roi_px)
    mask_result = build_shape_mask(median_roi, roi_px, logo_bbox, dilation=dilation)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mask_diagnostics(diagnostics_dir, mask_result)

    input_stem = video_path.stem
    telea_path = output_dir / f"{input_stem}_telea.mp4"
    ns_path = output_dir / f"{input_stem}_ns.mp4"
    encoders = {
        "telea": _start_encoder(video_path, telea_path, metadata),
        "ns": _start_encoder(video_path, ns_path, metadata),
    }
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        for process in encoders.values():
            process.kill()
        raise RuntimeError(f"OpenCV could not open {video_path}")

    representative = set(select_frame_indices(metadata.frame_count, sample_count))
    x, y, width, height = roi_px
    comparisons: list[np.ndarray] = []
    labels: list[str] = []
    method_seconds = {"telea": 0.0, "ns": 0.0}
    temporal_mad: dict[str, list[float]] = {"original": [], "telea": [], "ns": []}
    ring_temporal_mad: list[float] = []
    ring = cv2.dilate(mask_result.mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask_result.mask))
    likeness: dict[str, list[float]] = {"original": [], "telea": [], "ns": []}
    previous_rois: dict[str, np.ndarray | None] = {"telea": None, "ns": None}
    previous_original: np.ndarray | None = None
    worst_transition: dict[str, tuple[float, int]] = {"telea": (-1.0, -1), "ns": (-1.0, -1)}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            original_roi = frame[y:y + height, x:x + width].copy()
            if previous_original is not None:
                temporal_mad["original"].append(
                    _masked_temporal_mad(previous_original, original_roi, mask_result.mask)
                )
                ring_temporal_mad.append(_masked_temporal_mad(previous_original, original_roi, ring))
            restored: dict[str, np.ndarray] = {}
            for method in ("telea", "ns"):
                method_started = time.perf_counter()
                restored[method] = inpaint_roi(original_roi, mask_result.mask, method, inpaint_radius)
                method_seconds[method] += time.perf_counter() - method_started
                previous = previous_rois[method]
                if previous is not None:
                    mad = _masked_temporal_mad(previous, restored[method], mask_result.mask)
                    temporal_mad[method].append(mad)
                    if mad > worst_transition[method][0]:
                        worst_transition[method] = (mad, frame_index)
                previous_rois[method] = restored[method]
            previous_original = original_roi

            likeness["original"].append(_logo_likeness(original_roi, mask_result.raw_mask))
            likeness["telea"].append(_logo_likeness(restored["telea"], mask_result.raw_mask))
            likeness["ns"].append(_logo_likeness(restored["ns"], mask_result.raw_mask))

            for method, process in encoders.items():
                output_frame = frame.copy()
                output_frame[y:y + height, x:x + width] = restored[method]
                if process.stdin is None:
                    raise RuntimeError(f"FFmpeg {method} stdin is unavailable")
                process.stdin.write(output_frame.tobytes())

            if frame_index in representative:
                overlay = mask_overlay(original_roi, mask_result.mask)
                comparison = make_contact_sheet(
                    [original_roi, overlay, restored["telea"], restored["ns"]],
                    ["Original", "Mask overlay", "TELEA", "Navier-Stokes"],
                    columns=4,
                )
                write_image(diagnostics_dir / "comparisons" / f"frame_{frame_index:06d}.png", comparison)
                comparisons.append(comparison)
                labels.append(f"frame {frame_index} | {frame_index / metadata.fps:.3f}s")
            frame_index += 1
    except Exception:
        for process in encoders.values():
            process.kill()
        raise
    finally:
        capture.release()

    for method, process in encoders.items():
        _finish_encoder(process, method)
    if frame_index != metadata.frame_count:
        raise RuntimeError(f"Decoded {frame_index} frames, expected {metadata.frame_count}")

    write_image(diagnostics_dir / "comparison_contact_sheet.png", make_contact_sheet(comparisons, labels, columns=1))
    output_metadata = {"telea": asdict(probe_video(telea_path)), "ns": asdict(probe_video(ns_path))}
    report: dict[str, object] = {
        "experiment": 1,
        "cpu_only": True,
        "input": asdict(metadata),
        "roi_relative": asdict(relative_roi),
        "roi_pixels": dict(zip(("x", "y", "width", "height"), roi_px, strict=True)),
        "visible_logo_bbox_pixels": dict(zip(("x", "y", "width", "height"), logo_bbox, strict=True)),
        "mask": {
            "source_frames": frames_used_for_mask,
            "components": mask_result.component_count,
            "raw_pixels": mask_result.raw_pixel_count,
            "final_pixels": mask_result.final_pixel_count,
            "dilation_pixels": dilation,
            "coverage_of_roi": round(mask_result.final_pixel_count / (width * height), 6),
            "temporal_threshold": {"saturation_max": 120, "value_min": 100},
        },
        "inpaint_radius": inpaint_radius,
        "processing_seconds": {
            "telea_inpaint_only": round(method_seconds["telea"], 6),
            "ns_inpaint_only": round(method_seconds["ns"], 6),
            "total_including_mask_and_encoding": round(time.perf_counter() - started, 6),
        },
        "metrics": {
            "mean_logo_likeness": {key: round(float(np.mean(value)), 6) for key, value in likeness.items()},
            "mean_masked_temporal_mad": {
                key: round(float(np.mean(value)), 6) for key, value in temporal_mad.items()
            },
            "mean_context_ring_temporal_mad": round(float(np.mean(ring_temporal_mad)), 6),
            "masked_to_context_temporal_ratio": {
                key: round(float(np.mean(temporal_mad[key]) / np.mean(ring_temporal_mad)), 6)
                for key in ("telea", "ns")
            },
            "worst_masked_transition": {
                key: {"to_frame": value[1], "mad": round(value[0], 6)}
                for key, value in worst_transition.items()
            },
        },
        "outputs": {
            "telea": {"path": str(telea_path), "metadata": output_metadata["telea"]},
            "ns": {"path": str(ns_path), "metadata": output_metadata["ns"]},
        },
        "representative_frames": sorted(representative),
    }
    (diagnostics_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
