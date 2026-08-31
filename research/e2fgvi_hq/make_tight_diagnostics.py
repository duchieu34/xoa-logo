from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from research.e2fgvi_hq.benchmark import (
    BASELINE_VIDEOS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO,
    _read_segment,
)
from veo_watermark_remover.config import DEFAULT_CANDIDATE_ROI
from veo_watermark_remover.diagnostics import make_contact_sheet, write_image
from veo_watermark_remover.lama_cpu import context_crop_box
from veo_watermark_remover.video_io import probe_video
from veo_watermark_remover.watermark import load_measurement, scaled_logo_bbox


def main() -> int:
    parser = argparse.ArgumentParser(description="Render tight ROI diagnostics from saved benchmark")
    parser.add_argument("--start-frame", type=int, default=108)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    metadata = probe_video(DEFAULT_VIDEO)
    roi_x, roi_y, roi_width, roi_height = DEFAULT_CANDIDATE_ROI.to_pixels(
        metadata.width, metadata.height
    )
    logo_bbox = scaled_logo_bbox(metadata.width, metadata.height, load_measurement())
    crop = context_crop_box(metadata.width, metadata.height, logo_bbox, 256)
    local_roi = (roi_x - crop.x, roi_y - crop.y, roi_width, roi_height)

    sources = {"original": DEFAULT_VIDEO, **BASELINE_VIDEOS}
    sequences: dict[str, np.ndarray] = {}
    for name, path in sources.items():
        frames = _read_segment(path, args.start_frame, args.frames)
        sequences[name] = frames[:, roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    e2_path = args.output_dir / "e2fgvi_hq_crop.mp4"
    e2_frames = _read_segment(e2_path, 0, args.frames)
    x, y, width, height = local_roi
    sequences["e2fgvi_hq"] = e2_frames[:, y : y + height, x : x + width]

    frames_to_render = [128, 130, 131, 132, 133, 134, 136, 144]
    for absolute in frames_to_render:
        if not args.start_frame <= absolute < args.start_frame + args.frames:
            continue
        local = absolute - args.start_frame
        names = ("original", "telea", "alpha", "lama", "e2fgvi_hq")
        sheet = make_contact_sheet(
            [sequences[name][local] for name in names],
            ["Original", "TELEA", "Alpha-only", "LaMa", "E2FGVI-HQ"],
            columns=5,
        )
        write_image(args.output_dir / "tight_roi" / f"frame_{absolute:06d}.png", sheet)

    transition_names = ("telea", "alpha", "lama", "e2fgvi_hq")
    transition_images: list[np.ndarray] = []
    transition_labels: list[str] = []
    for name in transition_names:
        for absolute in range(130, 135):
            local = absolute - args.start_frame
            transition_images.append(sequences[name][local])
            transition_labels.append(f"{name} f{absolute}")
    write_image(
        args.output_dir / "tight_roi" / "transition_130_134.png",
        make_contact_sheet(transition_images, transition_labels, columns=5),
    )
    e2_transition = [
        sequences["e2fgvi_hq"][absolute - args.start_frame]
        for absolute in range(127, 136)
    ]
    write_image(
        args.output_dir / "tight_roi" / "e2fgvi_sequence_127_135.png",
        make_contact_sheet(
            e2_transition,
            [f"E2FGVI f{absolute}" for absolute in range(127, 136)],
            columns=9,
        ),
    )
    print(args.output_dir / "tight_roi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
