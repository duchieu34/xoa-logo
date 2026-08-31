from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_CANDIDATE_ROI
from .diagnostics import make_contact_sheet, write_image
from .mask import (
    MaskCalibrationError,
    build_shape_mask,
    collect_temporal_median,
    mask_overlay,
    save_mask_diagnostics,
)
from .video_io import probe_video
from .watermark import load_measurement, scaled_logo_bbox


def _read_roi_frames(
    video_path: Path,
    roi_px: tuple[int, int, int, int],
    frame_indices: list[int],
) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    x, y, width, height = roi_px
    frames: dict[int, np.ndarray] = {}
    try:
        for index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index} from {video_path}")
            frames[index] = frame[y : y + height, x : x + width].copy()
    finally:
        capture.release()
    return frames


def _sample_indices(frame_count: int, count: int = 8) -> list[int]:
    if frame_count < 1:
        return []
    return sorted(
        {
            round(value)
            for value in np.linspace(0, frame_count - 1, min(count, frame_count))
        }
    )


def calibrate_video_mask(
    video_path: Path,
    output_dir: Path,
    dilation: int = 1,
) -> dict[str, object]:
    metadata = probe_video(video_path)
    roi_px = DEFAULT_CANDIDATE_ROI.to_pixels(metadata.width, metadata.height)
    logo_bbox = scaled_logo_bbox(metadata.width, metadata.height, load_measurement())
    median_roi, decoded_frames = collect_temporal_median(video_path, roi_px)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_image(output_dir / "temporal_median_roi.png", median_roi)

    try:
        result = build_shape_mask(
            median_roi, roi_px, logo_bbox, dilation=dilation
        )
    except MaskCalibrationError as error:
        report: dict[str, object] = {
            "accepted": False,
            "video": asdict(metadata),
            "source": str(video_path),
            "roi": list(roi_px),
            "logo_bbox": list(logo_bbox),
            "decoded_frames": decoded_frames,
            "error": str(error),
            "calibration": error.details,
            "e2fgvi_allowed": False,
        }
        (output_dir / "calibration_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return report

    save_mask_diagnostics(output_dir, result)
    indices = _sample_indices(metadata.frame_count)
    roi_frames = _read_roi_frames(video_path, roi_px, indices)
    sheet_images: list[np.ndarray] = []
    sheet_labels: list[str] = []
    for index in indices:
        roi = roi_frames[index]
        raw_overlay = mask_overlay(roi, result.raw_mask)
        overlay = mask_overlay(roi, result.mask)
        write_image(output_dir / "frames" / f"frame_{index:06d}_roi.png", roi)
        write_image(
            output_dir / "frames" / f"frame_{index:06d}_raw_overlay.png",
            raw_overlay,
        )
        write_image(output_dir / "frames" / f"frame_{index:06d}_overlay.png", overlay)
        sheet_images.extend([roi, raw_overlay, overlay])
        sheet_labels.extend(
            [
                f"frame {index} original",
                f"frame {index} raw evidence",
                f"frame {index} refined final",
            ]
        )
    write_image(
        output_dir / "multi_frame_mask_overlay.png",
        make_contact_sheet(sheet_images, sheet_labels, columns=3),
    )
    zoom = 12
    write_image(
        output_dir / "mask_raw_zoom.png",
        cv2.resize(
            result.raw_mask,
            None,
            fx=zoom,
            fy=zoom,
            interpolation=cv2.INTER_NEAREST,
        ),
    )
    write_image(
        output_dir / "mask_final_zoom.png",
        cv2.resize(
            result.mask,
            None,
            fx=zoom,
            fy=zoom,
            interpolation=cv2.INTER_NEAREST,
        ),
    )

    local_x, local_y, logo_width, logo_height = result.local_logo_bbox
    report = {
        "accepted": True,
        "video": asdict(metadata),
        "source": str(video_path),
        "roi": list(roi_px),
        "logo_bbox": list(logo_bbox),
        "local_logo_bbox": [local_x, local_y, logo_width, logo_height],
        "decoded_frames": decoded_frames,
        "sample_frames": indices,
        "calibration": {
            "selection_strategy": result.selection_strategy,
            "detected_component_count": result.detected_component_count,
            "selected_component_count": result.component_count,
            "raw_mask_pixel_count": result.raw_pixel_count,
            "final_mask_pixel_count": result.final_pixel_count,
            "coverage_ratio": result.coverage_ratio,
            "span_ratio": list(result.span_ratio),
            "rectangularity": result.rectangularity,
            "confidence": result.confidence,
            "alignment_scale": result.alignment_scale,
            "alignment_offset": result.alignment_offset,
            "requested_dilation": result.requested_dilation,
            "effective_dilation": result.effective_dilation,
            "final_coverage_ratio": result.final_coverage_ratio,
        },
        "e2fgvi_allowed": False,
        "manual_review_required": True,
        "note": "Calibration passed numeric gates; approve overlays before E2FGVI.",
    }
    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate and inspect a per-video Veo mask without running removal"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dilation", type=int, default=1, choices=range(0, 5))
    args = parser.parse_args()
    report = calibrate_video_mask(args.video, args.output_dir, args.dilation)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
