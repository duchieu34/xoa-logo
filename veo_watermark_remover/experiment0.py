from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .config import RelativeROI
from .diagnostics import draw_roi, make_contact_sheet, save_report, write_image
from .video_io import VideoMetadata, probe_video


def select_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("Video frame count is unavailable or zero")
    count = min(sample_count, frame_count)
    # Include near-end content while avoiding duplicate integer indices.
    return sorted(set(np.linspace(0, frame_count - 1, count, dtype=int).tolist()))


def _roi_statistics(roi: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return {
        "mean_luma": round(float(gray.mean()), 3),
        "luma_stddev": round(float(gray.std()), 3),
        "mean_saturation": round(float(hsv[..., 1].mean()), 3),
        "edge_density": round(float((cv2.Canny(gray, 80, 160) > 0).mean()), 6),
    }


def run_experiment0(
    video_path: Path,
    output_dir: Path,
    relative_roi: RelativeROI,
    sample_count: int = 8,
) -> tuple[VideoMetadata, Path]:
    started = time.perf_counter()
    metadata = probe_video(video_path)
    roi_px = relative_roi.to_pixels(metadata.width, metadata.height)
    indices = select_frame_indices(metadata.frame_count, sample_count)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    overlays: list[np.ndarray] = []
    crops: list[np.ndarray] = []
    labels: list[str] = []
    records: list[dict[str, object]] = []
    x, y, width, height = roi_px
    try:
        for ordinal, index in enumerate(indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index}")
            crop = frame[y:y + height, x:x + width].copy()
            overlay = draw_roi(frame, roi_px)
            timestamp = index / metadata.fps if metadata.fps else 0.0
            stem = f"frame_{ordinal:02d}_{index:06d}"
            write_image(output_dir / "frames" / f"{stem}.png", frame)
            write_image(output_dir / "overlays" / f"{stem}_roi.png", overlay)
            write_image(output_dir / "rois" / f"{stem}_crop.png", crop)
            overlays.append(overlay)
            crops.append(crop)
            labels.append(f"frame {index} | {timestamp:.3f}s")
            records.append({
                "frame_index": index,
                "timestamp_seconds": round(timestamp, 6),
                "roi_statistics": _roi_statistics(crop),
            })
    finally:
        capture.release()

    stack = np.stack(crops).astype(np.float32)
    median_roi = np.median(stack, axis=0).astype(np.uint8)
    temporal_std = np.mean(np.std(stack, axis=0), axis=2)
    std_visual = cv2.applyColorMap(np.clip(temporal_std * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    write_image(output_dir / "roi_contact_sheet.png", make_contact_sheet(crops, labels))
    write_image(output_dir / "frame_contact_sheet.png", make_contact_sheet(overlays, labels, columns=2))
    write_image(output_dir / "median_roi.png", median_roi)
    write_image(output_dir / "temporal_std_roi.png", std_visual)
    save_report(
        output_dir / "report.json", metadata, relative_roi, roi_px, records,
        time.perf_counter() - started,
    )
    return metadata, output_dir

