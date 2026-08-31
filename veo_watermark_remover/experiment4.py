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
from .mask import build_shape_mask
from .optical_flow import (
    BILATERAL_FLOW_CODE,
    FLOW_REJECT_CONFIDENCE,
    FLOW_REJECT_CONTEXT,
    FLOW_REJECT_FORWARD_BACKWARD,
    FLOW_REJECT_NO_CLEAN_DONOR,
    estimate_dis_pair,
    reconstruct_optical_flow_temporal,
    visualize_flow,
)
from .temporal import reconstruct_direct_temporal, visualize_donor_source
from .video_io import probe_video
from .watermark import load_measurement, scaled_logo_bbox


def run_experiment4(
    video_path: Path,
    output_dir: Path,
    diagnostics_dir: Path,
    relative_roi: RelativeROI,
    sample_count: int = 8,
) -> dict[str, object]:
    started = time.perf_counter()
    metadata = probe_video(video_path)
    roi_px = relative_roi.to_pixels(metadata.width, metadata.height)
    x, y, width, height = roi_px
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

    direct = reconstruct_direct_temporal(
        roi_stack, alpha_array, alpha_valid, mask_result.mask, alpha_model.confidence
    )
    preflow_valid = alpha_valid | (direct.donor_mask > 0)
    flow_started = time.perf_counter()
    flow = reconstruct_optical_flow_temporal(
        roi_stack,
        direct.restored_stack,
        preflow_valid,
        mask_result.mask,
        alpha_model.confidence,
        flow_estimator=estimate_dis_pair,
    )
    flow_seconds = time.perf_counter() - flow_started

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    donor = flow.donor_mask > 0
    donor_counts_per_frame = np.count_nonzero(donor, axis=(1, 2))
    donor_frequency = np.count_nonzero(donor, axis=0).astype(np.float32)
    if donor_frequency.max() > 0:
        donor_frequency /= donor_frequency.max()
    write_image(
        diagnostics_dir / "flow_donor_frequency_map.png",
        visualize_scalar(donor_frequency, mask_result.mask),
    )
    write_image(
        diagnostics_dir / "flow_confidence_max_map.png",
        visualize_scalar(np.max(flow.confidence, axis=0), mask_result.mask),
    )
    write_image(
        diagnostics_dir / "always_unresolved_mask.png",
        np.all(flow.unresolved_stack > 0, axis=0).astype(np.uint8) * 255,
    )

    representative = set(select_frame_indices(metadata.frame_count, sample_count))
    nonzero_donor_frames = np.flatnonzero(donor_counts_per_frame)
    top_count = min(8, nonzero_donor_frames.size)
    top_donor_frames = set(
        nonzero_donor_frames[np.argsort(donor_counts_per_frame[nonzero_donor_frames])[-top_count:]].tolist()
    )
    known_difficult = {index for index in (128, 129, 130, 131) if index < metadata.frame_count}
    diagnostic_frames = sorted(representative | top_donor_frames | known_difficult)
    comparisons: list[np.ndarray] = []
    labels: list[str] = []
    telea_stack: list[np.ndarray] = []
    for frame_index, original_roi in enumerate(roi_stack):
        telea_roi = inpaint_roi(original_roi, mask_result.mask, "telea", 3)
        telea_stack.append(telea_roi)
        if frame_index not in diagnostic_frames:
            continue
        source_visual = visualize_donor_source(flow.donor_source[frame_index])
        comparison = make_contact_sheet(
            [
                original_roi,
                telea_roi,
                alpha_array[frame_index],
                direct.restored_stack[frame_index],
                flow.restored_stack[frame_index],
                source_visual,
            ],
            ["Original", "TELEA", "Alpha", "Direct temporal", "Optical Flow", "Flow donor"],
            columns=6,
        )
        write_image(diagnostics_dir / "comparisons" / f"frame_{frame_index:06d}.png", comparison)
        write_image(
            diagnostics_dir / "maps" / f"frame_{frame_index:06d}_donor_source.png",
            source_visual,
        )
        write_image(
            diagnostics_dir / "maps" / f"frame_{frame_index:06d}_confidence.png",
            visualize_scalar(flow.confidence[frame_index], mask_result.mask),
        )
        write_image(
            diagnostics_dir / "maps" / f"frame_{frame_index:06d}_flow.png",
            visualize_flow(flow.best_flow[frame_index], donor[frame_index]),
        )
        fb_visual = np.zeros(mask_result.mask.shape, dtype=np.float32)
        finite = donor[frame_index] & np.isfinite(flow.best_forward_backward_error[frame_index])
        fb_visual[finite] = np.clip(flow.best_forward_backward_error[frame_index][finite], 0, 2) / 2
        write_image(
            diagnostics_dir / "maps" / f"frame_{frame_index:06d}_fb_error.png",
            visualize_scalar(fb_visual, mask_result.mask),
        )
        write_image(
            diagnostics_dir / "maps" / f"frame_{frame_index:06d}_unresolved.png",
            flow.unresolved_stack[frame_index],
        )
        if frame_index in representative:
            comparisons.append(comparison)
            labels.append(
                f"frame {frame_index} | {frame_index / metadata.fps:.3f}s | "
                f"flow donors {donor_counts_per_frame[frame_index]}"
            )
    telea_array = np.stack(telea_stack)
    write_image(
        diagnostics_dir / "comparison_contact_sheet.png",
        make_contact_sheet(comparisons, labels, columns=1),
    )

    output_path = output_dir / f"{video_path.stem}_optical_flow.mp4"
    encoder = _start_encoder(video_path, output_path, metadata)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        encoder.kill()
        raise RuntimeError(f"OpenCV could not open {video_path}")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame[y:y + height, x:x + width] = flow.restored_stack[frame_index]
            if encoder.stdin is None:
                raise RuntimeError("FFmpeg optical-flow encoder stdin is unavailable")
            encoder.stdin.write(frame.tobytes())
            frame_index += 1
    except Exception:
        encoder.kill()
        raise
    finally:
        capture.release()
    _finish_encoder(encoder, "optical-flow")
    output_metadata = probe_video(output_path)

    sequences = {
        "original": roi_stack,
        "telea": telea_array,
        "alpha": alpha_array,
        "direct_temporal": direct.restored_stack,
        "optical_flow": flow.restored_stack,
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

    static_unresolved = alpha_model.unresolved_mask > 0
    raw_logo = mask_result.raw_mask > 0
    source_counts = Counter(map(int, flow.donor_source[donor].tolist()))
    accepted_flow = flow.best_flow[donor]
    accepted_magnitude = np.linalg.norm(accepted_flow, axis=1) if accepted_flow.size else np.array([])
    accepted_fb = flow.best_forward_backward_error[donor]
    eligible_candidate = flow.best_candidate_confidence > 0
    rejection = flow.rejection_reason
    rescued_static_positions = np.any(donor, axis=0) & static_unresolved
    events: list[dict[str, object]] = []
    for event_frame, event_y, event_x in np.argwhere(donor):
        before = direct.restored_stack[event_frame, event_y, event_x]
        after = flow.restored_stack[event_frame, event_y, event_x]
        events.append({
            "frame": int(event_frame),
            "roi_xy": [int(event_x), int(event_y)],
            "frame_xy": [int(x + event_x), int(y + event_y)],
            "source_code": int(flow.donor_source[event_frame, event_y, event_x]),
            "confidence": round(float(flow.confidence[event_frame, event_y, event_x]), 6),
            "flow_xy": [
                round(float(flow.best_flow[event_frame, event_y, event_x, 0]), 6),
                round(float(flow.best_flow[event_frame, event_y, event_x, 1]), 6),
            ],
            "forward_backward_error": round(
                float(flow.best_forward_backward_error[event_frame, event_y, event_x]), 6
            ),
            "before_bgr": list(map(int, before)),
            "after_bgr": list(map(int, after)),
            "inside_raw_logo_mask": bool(raw_logo[event_y, event_x]),
        })
    report: dict[str, object] = {
        "experiment": 4,
        "cpu_only": True,
        "prohibited_methods_used": {
            "gpu_or_cuda": False,
            "inpaint_for_output": False,
            "blur_or_patch": False,
        },
        "input": asdict(metadata),
        "roi_pixels": dict(zip(("x", "y", "width", "height"), roi_px, strict=True)),
        "method": {
            "flow": "OpenCV DIS medium preset, CPU only, on ROI",
            "offsets": [-3, -2, -1, 1, 2, 3],
            "logo_flow_from_surrounding_context": True,
            "source_must_be_outside_unresolved_watermark": True,
            "forward_backward_max_pixels": 1.0,
            "context_radius": 5,
            "bilateral_agreement_max": 16.0,
            "bilateral_confidence_min": 0.15,
            "single_confidence_min": 0.25,
            "bilateral_preferred": True,
            "processing_order": ["alpha", "direct_temporal", "optical_flow", "unresolved"],
        },
        "coverage": {
            "mask_pixels": int(np.count_nonzero(mask_result.mask)),
            "experiment2_static_unresolved_pixels": int(np.count_nonzero(static_unresolved)),
            "experiment3_direct_donor_pixel_frames": int(np.count_nonzero(direct.donor_mask)),
            "flow_donor_pixel_frames": int(np.count_nonzero(donor)),
            "eligible_candidate_pixel_frames_before_confidence": int(
                np.count_nonzero(eligible_candidate)
            ),
            "flow_unique_pixel_positions": int(np.count_nonzero(np.any(donor, axis=0))),
            "frames_with_flow_donor": int(np.count_nonzero(donor_counts_per_frame)),
            "rescued_static_unresolved_pixel_frames": int(np.count_nonzero(donor & static_unresolved[None])),
            "rescued_static_unresolved_unique_pixels": int(np.count_nonzero(rescued_static_positions)),
            "rescued_raw_logo_pixel_frames": int(np.count_nonzero(donor & raw_logo[None])),
            "remaining_unresolved_pixel_frames": int(np.count_nonzero(flow.unresolved_stack)),
            "remaining_always_unresolved_pixels": int(np.count_nonzero(np.all(flow.unresolved_stack > 0, axis=0))),
            "flow_donors_per_frame": _quantiles(donor_counts_per_frame.astype(np.float32)),
            "remaining_unresolved_per_frame": _quantiles(
                np.count_nonzero(flow.unresolved_stack, axis=(1, 2)).astype(np.float32)
            ),
        },
        "donors": {
            "source_code_counts": {str(key): value for key, value in sorted(source_counts.items())},
            "bilateral_code": BILATERAL_FLOW_CODE,
            "confidence_quantiles": _quantiles(flow.confidence[donor]),
            "flow_magnitude_quantiles": _quantiles(accepted_magnitude.astype(np.float32)),
            "forward_backward_error_quantiles": _quantiles(accepted_fb.astype(np.float32)),
            "eligible_candidate_confidence_quantiles": _quantiles(
                flow.best_candidate_confidence[eligible_candidate]
            ),
            "top_frames": [
                {"frame": int(index), "donor_pixels": int(donor_counts_per_frame[index])}
                for index in sorted(top_donor_frames, key=lambda item: donor_counts_per_frame[item], reverse=True)
            ],
            "events": events,
        },
        "rejections_by_pixel_frame": {
            "no_clean_warped_donor": int(np.count_nonzero(rejection == FLOW_REJECT_NO_CLEAN_DONOR)),
            "forward_backward_inconsistent": int(np.count_nonzero(rejection == FLOW_REJECT_FORWARD_BACKWARD)),
            "context_mismatch": int(np.count_nonzero(rejection == FLOW_REJECT_CONTEXT)),
            "confidence_or_consensus": int(np.count_nonzero(rejection == FLOW_REJECT_CONFIDENCE)),
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
            "optical_flow_roi_only": round(flow_seconds, 6),
            "total_including_alpha_direct_diagnostics_and_encoding": round(
                time.perf_counter() - started, 6
            ),
        },
        "output": {"path": str(output_path), "metadata": asdict(output_metadata)},
        "representative_frames": sorted(representative),
        "warning": "Confidence-gated flow experiment; unresolved pixels remain unchanged.",
    }
    (diagnostics_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
