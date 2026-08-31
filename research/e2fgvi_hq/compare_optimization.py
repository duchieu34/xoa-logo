from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.e2fgvi_hq.benchmark import DEFAULT_OUTPUT_DIR, DEFAULT_VIDEO, _read_segment
from research.e2fgvi_hq.smoke_test import WORKSPACE
from veo_watermark_remover.config import DEFAULT_CANDIDATE_ROI
from veo_watermark_remover.diagnostics import make_contact_sheet, write_image
from veo_watermark_remover.lama_cpu import context_crop_box
from veo_watermark_remover.video_io import probe_video
from veo_watermark_remover.watermark import load_measurement, scaled_logo_bbox


RESULTS_DIR = WORKSPACE / "research" / "e2fgvi_hq" / "results" / "optimization"
VARIANTS = {
    "control256": (
        RESULTS_DIR / "crop_256_control.json",
        DEFAULT_OUTPUT_DIR / "optimization/crop_256_control",
        256,
    ),
    "crop192": (RESULTS_DIR / "crop_192.json", DEFAULT_OUTPUT_DIR / "optimization/crop_192", 192),
    "crop224": (RESULTS_DIR / "crop_224.json", DEFAULT_OUTPUT_DIR / "optimization/crop_224", 224),
    "ref20": (RESULTS_DIR / "ref_step_20.json", DEFAULT_OUTPUT_DIR / "optimization/ref_step_20", 256),
    "ref30": (RESULTS_DIR / "ref_step_30.json", DEFAULT_OUTPUT_DIR / "optimization/ref_step_30", 256),
    "weighted": (RESULTS_DIR / "weighted_256.json", DEFAULT_OUTPUT_DIR / "optimization/weighted_256", 256),
    "final_best": (
        RESULTS_DIR / "final_best.json",
        DEFAULT_OUTPUT_DIR / "optimization/final_best",
        192,
    ),
}


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    metric = report["metrics"]["e2fgvi_hq"]
    return {
        "seconds_per_frame": report["runtime"]["inference_seconds_per_output_frame"],
        "peak_rss_mb": report["runtime"]["peak_rss_mb"],
        "logo_likeness": metric["mean_logo_likeness"],
        "temporal_mad": metric["mean_masked_temporal_mad"],
        "mad_130_131": metric["transition_130_to_131_mad"],
        "mad_132_133": metric.get("transition_132_to_133_mad"),
        "luma_delta_132_133": metric.get("transition_132_to_133_mean_luma_delta"),
        "laplacian": metric.get("mean_raw_mask_laplacian_energy"),
        "padded": [
            report["crop"]["hq_internal_padded_height"],
            report["crop"]["hq_internal_padded_width"],
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare E2FGVI optimization variants")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "optimization/comparison"
    )
    args = parser.parse_args()
    metadata = probe_video(DEFAULT_VIDEO)
    roi_x, roi_y, roi_width, roi_height = DEFAULT_CANDIDATE_ROI.to_pixels(
        metadata.width, metadata.height
    )
    logo_bbox = scaled_logo_bbox(metadata.width, metadata.height, load_measurement())
    sequences = {}
    summary = {}
    for name, (report_path, output_dir, crop_size) in VARIANTS.items():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary[name] = _compact(report)
        crop = context_crop_box(metadata.width, metadata.height, logo_bbox, crop_size)
        frames = _read_segment(output_dir / "e2fgvi_hq_crop.mp4", 0, 48)
        local_x, local_y = roi_x - crop.x, roi_y - crop.y
        sequences[name] = frames[
            :, local_y : local_y + roi_height, local_x : local_x + roi_width
        ]

    names = tuple(VARIANTS)
    for absolute in (128, 130, 131, 132, 133, 134, 136, 144):
        local = absolute - 108
        write_image(
            args.output_dir / f"frame_{absolute:06d}.png",
            make_contact_sheet(
                [sequences[name][local] for name in names], list(names), columns=3
            ),
        )
    write_image(
        args.output_dir / "sequence_130_134.png",
        make_contact_sheet(
            [sequences[name][absolute - 108] for name in names for absolute in range(130, 135)],
            [f"{name} f{absolute}" for name in names for absolute in range(130, 135)],
            columns=5,
        ),
    )
    (RESULTS_DIR / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
