from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CANDIDATE_ROI, RelativeROI
from .experiment0 import run_experiment0
from .experiment1 import run_experiment1
from .experiment2 import run_experiment2
from .experiment3 import run_experiment3
from .experiment4 import run_experiment4
from .ai_experiment1 import run_ai_experiment1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only Veo watermark research experiments."
    )
    parser.add_argument("input", type=Path, help="Input video (typically MP4)")
    parser.add_argument(
        "--diagnostics", "-d", type=Path, default=None,
        help="Diagnostic output directory",
    )
    parser.add_argument(
        "--roi", default=None, metavar="X,Y,W,H",
        help="Relative candidate ROI; each value is in [0,1]",
    )
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to inspect (5-10 recommended)")
    parser.add_argument("--experiment", type=int, choices=(0, 1, 2, 3, 4), default=0, help="Experiment to run")
    parser.add_argument("--ai-experiment", type=int, choices=(1,), default=None, help="Run a separate AI experiment")
    parser.add_argument("--lama-model", type=Path, default=Path("models/big-lama.pt"), help="Path to big-LaMa TorchScript model")
    parser.add_argument("--lama-context", type=int, choices=(192, 256), default=256, help="Selected LaMa context crop size")
    parser.add_argument("--output-dir", type=Path, default=None, help="Experiment video output directory")
    parser.add_argument("--mask-dilation", type=int, default=1, help="Experiment 1 anti-alias mask dilation in pixels")
    parser.add_argument("--inpaint-radius", type=float, default=3.0, help="Experiment 1 OpenCV inpaint radius")
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
        diagnostics = args.diagnostics or (
            Path(f"diagnostics/ai_experiment{args.ai_experiment}")
            if args.ai_experiment is not None else Path(f"diagnostics/experiment{args.experiment}")
        )
        if args.ai_experiment == 1:
            output_dir = args.output_dir or Path("outputs/ai_experiment1")
            report = run_ai_experiment1(
                args.input, output_dir, diagnostics, roi, args.lama_model,
                args.frames, args.lama_context,
            )
            print(f"AI Experiment 1 complete: {report['input']['frame_count']} frames")
            print(f"Diagnostics: {diagnostics.resolve()}")
            print(f"Diagnostic video: {output_dir.resolve()}")
        elif args.experiment == 0:
            metadata, output_dir = run_experiment0(args.input, diagnostics, roi, args.frames)
            print(
                f"Experiment 0 complete: {metadata.width}x{metadata.height}, "
                f"{metadata.fps:.3f} FPS, {metadata.duration_seconds:.3f}s"
            )
            print(f"Diagnostics: {output_dir.resolve()}")
            print("No watermark removal was performed.")
        elif args.experiment == 1:
            output_dir = args.output_dir or Path("outputs/experiment1")
            report = run_experiment1(
                args.input, output_dir, diagnostics, roi, args.frames,
                args.mask_dilation, args.inpaint_radius,
            )
            print(f"Experiment 1 complete: {report['input']['frame_count']} frames")
            print(f"Diagnostics: {diagnostics.resolve()}")
            print(f"Videos: {output_dir.resolve()}")
        elif args.experiment == 2:
            output_dir = args.output_dir or Path("outputs/experiment2")
            report = run_experiment2(args.input, output_dir, diagnostics, roi, args.frames)
            print(f"Experiment 2 complete: {report['input']['frame_count']} frames")
            print(f"Diagnostics: {diagnostics.resolve()}")
            print(f"Diagnostic video: {output_dir.resolve()}")
        elif args.experiment == 3:
            output_dir = args.output_dir or Path("outputs/experiment3")
            report = run_experiment3(args.input, output_dir, diagnostics, roi, args.frames)
            print(f"Experiment 3 complete: {report['input']['frame_count']} frames")
            print(f"Diagnostics: {diagnostics.resolve()}")
            print(f"Diagnostic video: {output_dir.resolve()}")
        else:
            output_dir = args.output_dir or Path("outputs/experiment4")
            report = run_experiment4(args.input, output_dir, diagnostics, roi, args.frames)
            print(f"Experiment 4 complete: {report['input']['frame_count']} frames")
            print(f"Diagnostics: {diagnostics.resolve()}")
            print(f"Diagnostic video: {output_dir.resolve()}")
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
