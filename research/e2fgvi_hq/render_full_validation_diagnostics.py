from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from research.e2fgvi_hq.benchmark import BASELINE_VIDEOS, DEFAULT_VIDEO
from research.e2fgvi_hq.full_video_validation import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT,
    _read_crop_video,
)
from veo_watermark_remover.config import DEFAULT_CANDIDATE_ROI
from veo_watermark_remover.diagnostics import make_contact_sheet, write_image
from veo_watermark_remover.experiment1 import _masked_temporal_mad
from veo_watermark_remover.experiment2 import _collect_roi_stack
from veo_watermark_remover.lama_cpu import context_crop_box, mask_in_context
from veo_watermark_remover.mask import build_shape_mask
from veo_watermark_remover.video_io import probe_video
from veo_watermark_remover.watermark import load_measurement, scaled_logo_bbox


def _tight_box(mask: np.ndarray, margin: int = 24) -> tuple[int, int, int, int]:
    x, y, width, height = cv2.boundingRect((mask > 0).astype(np.uint8))
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(mask.shape[1], x + width + margin), min(mask.shape[0], y + height + margin)
    return x1, y1, x2 - x1, y2 - y1


def _cut(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = box
    return image[y : y + height, x : x + width]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render tight full-validation diagnostics")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    metadata = probe_video(DEFAULT_VIDEO)
    roi_px = DEFAULT_CANDIDATE_ROI.to_pixels(metadata.width, metadata.height)
    roi_stack = _collect_roi_stack(DEFAULT_VIDEO, roi_px)
    median_roi = np.median(roi_stack, axis=0).astype(np.uint8)
    logo_bbox = scaled_logo_bbox(metadata.width, metadata.height, load_measurement())
    mask_result = build_shape_mask(median_roi, roi_px, logo_bbox, dilation=1)
    crop = context_crop_box(metadata.width, metadata.height, logo_bbox, 192)
    mask = mask_in_context(mask_result.mask, roi_px, crop)
    tight = _tight_box(mask)
    full_output = args.output_dir / "ft-vid-23_e2fgvi_hq_cpu_full.mp4"
    paths = {
        "original": DEFAULT_VIDEO,
        "telea": BASELINE_VIDEOS["telea"],
        "lama": BASELINE_VIDEOS["lama"],
        "e2fgvi_hq": full_output,
    }
    sequences = {
        name: _read_crop_video(path, crop, metadata.frame_count) for name, path in paths.items()
    }
    names = tuple(paths)
    transitions = report["quality"]["top_10_e2fgvi_transitions"]
    for rank, transition in enumerate(transitions, start=1):
        before, after = int(transition["from_frame"]), int(transition["to_frame"])
        images = []
        labels = []
        for name in names:
            mad = _masked_temporal_mad(sequences[name][before], sequences[name][after], mask)
            images.extend([_cut(sequences[name][before], tight), _cut(sequences[name][after], tight)])
            labels.extend([f"{name} f{before}", f"{name} f{after} | MAD {mad:.2f}"])
        write_image(
            args.output_dir
            / "top_transitions_tight"
            / f"top_{rank:02d}_frame_{before:06d}_to_{after:06d}.png",
            make_contact_sheet(images, labels, columns=4),
        )

    darkest = report["quality"]["e2fgvi_dark_patch_proxy"]["darkest_frames"]
    for rank, record in enumerate(darkest, start=1):
        frame = int(record["frame"])
        write_image(
            args.output_dir / "dark_patch_frames" / f"dark_{rank:02d}_frame_{frame:06d}.png",
            make_contact_sheet(
                [_cut(sequences[name][frame], tight) for name in names],
                [f"{name} f{frame}" for name in names],
                columns=4,
            ),
        )
    for start, end in ((10, 15), (100, 103), (127, 136)):
        write_image(
            args.output_dir / "sequences" / f"e2fgvi_frames_{start:03d}_{end:03d}.png",
            make_contact_sheet(
                [_cut(sequences["e2fgvi_hq"][frame], tight) for frame in range(start, end + 1)],
                [f"E2FGVI f{frame}" for frame in range(start, end + 1)],
                columns=end - start + 1,
            ),
        )
    print(
        json.dumps(
            {"tight_box": tight, "top_transitions": len(transitions), "dark_frames": len(darkest)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
