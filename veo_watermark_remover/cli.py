from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CANDIDATE_ROI, RelativeROI
from .experiment0 import run_experiment0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 0: inspect a Veo-watermarked video without modifying it."
    )
    parser.add_argument("input", type=Path, help="Input video (typically MP4)")
    parser.add_argument(
        "--diagnostics", "-d", type=Path, default=Path("diagnostics/experiment0"),
        help="Diagnostic output directory",
    )
    parser.add_argument(
        "--roi", default=None, metavar="X,Y,W,H",
        help="Relative candidate ROI; each value is in [0,1]",
    )
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to inspect (5-10 recommended)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input video does not exist: {args.input}")
    if not 1 <= args.frames <= 100:
        parser.error("--frames must be between 1 and 100")
    try:
        roi = RelativeROI.parse(args.roi) if args.roi else DEFAULT_CANDIDATE_ROI
        metadata, output_dir = run_experiment0(args.input, args.diagnostics, roi, args.frames)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Experiment 0 complete: {metadata.width}x{metadata.height}, "
        f"{metadata.fps:.3f} FPS, {metadata.duration_seconds:.3f}s"
    )
    print(f"Diagnostics: {output_dir.resolve()}")
    print("No watermark removal was performed.")
    return 0

