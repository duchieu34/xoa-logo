from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .alpha_recovery import deblend_roi, estimate_distribution_model, visualize_scalar
from .config import RelativeROI
from .diagnostics import make_contact_sheet, write_image
from .experiment0 import select_frame_indices
from .experiment1 import _finish_encoder, _logo_likeness, _masked_temporal_mad, _start_encoder
from .experiment2 import _collect_roi_stack, _quantiles
from .inpaint import inpaint_roi
from .mask import build_shape_mask, mask_overlay
from .temporal import (
    BILATERAL_DONOR_CODE,
    REJECT_CONFIDENCE_OR_CONSENSUS,
    REJECT_CONTEXT_MISMATCH,
    REJECT_NO_CLEAN_SOURCE,
    reconstruct_direct_temporal,
    visualize_donor_source,
)
from .video_io import probe_video
from .watermark import load_measurement, scaled_logo_bbox


def run_experiment3(
    video_path: Path,
    output_dir: Path,
    diagnostics_dir: Path,
    relative_roi: RelativeROI,
    sample_count: int = 8,
) -> dict[str, object]:
    started = time.perf_counter()
    metadata = probe_video(video_path)
    roi_px = relative_roi.to_pixels(metadata.width, metadata.height)
    roi_stack = _collect_roi_stack(video_path, roi_px)
    median_roi = np.median(roi_stack, axis=0).astype(np.uint8)
    logo_bbox = scaled_logo_bbox(metadata.width, metadata.height, load_measurement())
    mask_result = build_shape_mask(median_roi, roi_px, logo_bbox, dilation=1)
    alpha_model, _ = estimate_distribution_model(roi_stack, mask_result.mask)

    alpha_stack: list[np.ndarray] = []
    alpha_valid_stack: list[np.ndarray] = []
    for roi in roi_stack:
        restored, valid = deblend_roi(roi, alpha_model)
        alpha_stack.append(restored)
        alpha_valid_stack.append(valid > 0)
    alpha_array = np.stack(alpha_stack)
    alpha_valid = np.stack(alpha_valid_stack)

    temporal_started = time.perf_counter()
    temporal = reconstruct_direct_temporal(
        roi_stack,
        alpha_array,
        alpha_valid,
        mask_result.mask,
        alpha_model.confidence,
    )
    temporal_seconds = time.perf_counter() - temporal_started

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    donor_frequency = np.count_nonzero(temporal.donor_mask, axis=0).astype(np.float32)
    if donor_frequency.max() > 0:
        donor_frequency /= donor_frequency.max()
    write_image(
        diagnostics_dir / "donor_frequency_map.png",
        visualize_scalar(donor_frequency, mask_result.mask),
    )
    write_image(
        diagnostics_dir / "temporal_confidence_max_map.png",
        visualize_scalar(np.max(temporal.confidence, axis=0), mask_result.mask),
    )
    write_image(
        diagnostics_dir / "always_unresolved_mask.png",
        np.all(temporal.unresolved_stack > 0, axis=0).astype(np.uint8) * 255,
    )

    representative = set(select_frame_indices(metadata.frame_count, sample_count))
    donor_frames = set(np.flatnonzero(np.any(temporal.donor_mask > 0, axis=(1, 2))).tolist())
    diagnostic_frames = sorted(representative | donor_frames)
    comparisons: list[np.ndarray] = []
    comparison_labels: list[str] = []
    telea_stack: list[np.ndarray] = []
    for frame_index, original_roi in enumerate(roi_stack):
        telea_roi = inpaint_roi(original_roi, mask_result.mask, "telea", 3)
        telea_stack.append(telea_roi)
        if frame_index in diagnostic_frames:
            source_visual = visualize_donor_source(temporal.donor_source[frame_index])
            comparison = make_contact_sheet(
                [
                    original_roi,
                    telea_roi,
                    alpha_array[frame_index],
                    temporal.restored_stack[frame_index],
                    source_visual,
                ],
                ["Original", "TELEA", "Alpha", "Direct temporal", "Donor source"],
                columns=5,
            )
            write_image(diagnostics_dir / "comparisons" / f"frame_{frame_index:06d}.png", comparison)
            write_image(
                diagnostics_dir / "maps" / f"frame_{frame_index:06d}_donor_source.png",
                source_visual,
            )
            write_image(
                diagnostics_dir / "maps" / f"frame_{frame_index:06d}_temporal_confidence.png",
                visualize_scalar(temporal.confidence[frame_index], mask_result.mask),
            )
            write_image(
                diagnostics_dir / "maps" / f"frame_{frame_index:06d}_unresolved.png",
                temporal.unresolved_stack[frame_index],
            )
            if frame_index in representative:
                comparisons.append(comparison)
                comparison_labels.append(f"frame {frame_index} | {frame_index / metadata.fps:.3f}s")
    telea_array = np.stack(telea_stack)
    write_image(
        diagnostics_dir / "comparison_contact_sheet.png",
        make_contact_sheet(comparisons, comparison_labels, columns=1),
    )

    output_path = output_dir / f"{video_path.stem}_direct_temporal.mp4"
    encoder = _start_encoder(video_path, output_path, metadata)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        encoder.kill()
        raise RuntimeError(f"OpenCV could not open {video_path}")
    x, y, width, height = roi_px
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame[y:y + height, x:x + width] = temporal.restored_stack[frame_index]
            if encoder.stdin is None:
                raise RuntimeError("FFmpeg direct-temporal encoder stdin is unavailable")
            encoder.stdin.write(frame.tobytes())
            frame_index += 1
    except Exception:
        encoder.kill()
        raise
    finally:
        capture.release()
    _finish_encoder(encoder, "direct-temporal")
    output_metadata = probe_video(output_path)

    sequences = {
        "original": roi_stack,
        "telea": telea_array,
        "alpha": alpha_array,
        "direct_temporal": temporal.restored_stack,
    }
    likeness: dict[str, float] = {}
    temporal_mad: dict[str, float] = {}
    worst_transition: dict[str, dict[str, float | int]] = {}
    for name, sequence in sequences.items():
        likeness[name] = float(np.mean([
            _logo_likeness(roi, mask_result.raw_mask) for roi in sequence
        ]))
        differences = [
            _masked_temporal_mad(sequence[index - 1], sequence[index], mask_result.mask)
            for index in range(1, len(sequence))
        ]
        worst_index = int(np.argmax(differences)) + 1
        temporal_mad[name] = float(np.mean(differences))
        worst_transition[name] = {"to_frame": worst_index, "mad": float(differences[worst_index - 1])}

    donor = temporal.donor_mask > 0
    static_unresolved = alpha_model.unresolved_mask > 0
    raw_logo = mask_result.raw_mask > 0
    dynamic_alpha_gaps = (mask_result.mask > 0)[None] & ~alpha_valid & ~static_unresolved[None]
    source_counts = Counter(map(int, temporal.donor_source[donor].tolist()))
    events: list[dict[str, object]] = []
    for event_frame, event_y, event_x in np.argwhere(donor):
        events.append({
            "frame": int(event_frame),
            "roi_xy": [int(event_x), int(event_y)],
            "frame_xy": [int(x + event_x), int(y + event_y)],
            "source_code": int(temporal.donor_source[event_frame, event_y, event_x]),
            "confidence": round(float(temporal.confidence[event_frame, event_y, event_x]), 6),
        })

    rejection = temporal.rejection_reason
    report: dict[str, object] = {
        "experiment": 3,
        "cpu_only": True,
        "prohibited_methods_used": {
            "optical_flow": False,
            "inpaint_for_output": False,
            "blur_or_patch": False,
        },
        "input": asdict(metadata),
        "roi_pixels": dict(zip(("x", "y", "width", "height"), roi_px, strict=True)),
        "method": {
            "offsets": [-3, -2, -1, 1, 2, 3],
            "same_coordinate_only": True,
            "source_must_be_alpha_valid": True,
            "context_radius": 4,
            "context_color_mad_max": 18.0,
            "context_gradient_mad_max": 24.0,
            "bilateral_agreement_max": 14.0,
            "bilateral_preferred": True,
        },
        "coverage": {
            "experiment2_static_resolved_pixels": int(np.count_nonzero(alpha_model.resolved_mask)),
            "experiment2_static_unresolved_pixels": int(np.count_nonzero(static_unresolved)),
            "experiment2_alpha_valid_pixel_frames": int(np.count_nonzero(alpha_valid)),
            "dynamic_alpha_gap_pixel_frames": int(np.count_nonzero(dynamic_alpha_gaps)),
            "temporal_donor_pixel_frames": int(np.count_nonzero(donor)),
            "temporal_unique_pixel_positions": int(np.count_nonzero(np.any(donor, axis=0))),
            "frames_with_temporal_donor": len(donor_frames),
            "rescued_from_301_static_unresolved": int(np.count_nonzero(donor & static_unresolved[None])),
            "rescued_dynamic_alpha_gaps": int(np.count_nonzero(donor & dynamic_alpha_gaps)),
            "rescued_raw_logo_pixel_frames": int(np.count_nonzero(donor & raw_logo[None])),
            "remaining_unresolved_pixel_frames": int(np.count_nonzero(temporal.unresolved_stack)),
            "remaining_unresolved_per_frame": _quantiles(
                np.count_nonzero(temporal.unresolved_stack, axis=(1, 2)).astype(np.float32)
            ),
        },
        "donors": {
            "source_code_counts": {str(key): value for key, value in sorted(source_counts.items())},
            "bilateral_code": BILATERAL_DONOR_CODE,
            "confidence_quantiles": _quantiles(temporal.confidence[donor]),
            "events": events,
        },
        "rejections_by_pixel_frame": {
            "no_clean_source": int(np.count_nonzero(rejection == REJECT_NO_CLEAN_SOURCE)),
            "context_mismatch": int(np.count_nonzero(rejection == REJECT_CONTEXT_MISMATCH)),
            "confidence_or_consensus": int(np.count_nonzero(rejection == REJECT_CONFIDENCE_OR_CONSENSUS)),
        },
        "metrics": {
            "mean_logo_likeness": {key: round(value, 6) for key, value in likeness.items()},
            "mean_masked_temporal_mad": {key: round(value, 6) for key, value in temporal_mad.items()},
            "worst_masked_transition": {
                key: {"to_frame": value["to_frame"], "mad": round(float(value["mad"]), 6)}
                for key, value in worst_transition.items()
            },
        },
        "processing_seconds": {
            "direct_temporal_roi_only": round(temporal_seconds, 6),
            "total_including_alpha_model_diagnostics_and_encoding": round(time.perf_counter() - started, 6),
        },
        "output": {"path": str(output_path), "metadata": asdict(output_metadata)},
        "representative_frames": sorted(representative),
        "warning": "Direct same-coordinate baseline; no spatial motion compensation or optical flow.",
    }
    (diagnostics_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report

