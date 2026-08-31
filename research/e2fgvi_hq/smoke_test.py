from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from research.e2fgvi_hq.mmcv_cpu_shim import install_mmcv_cpu_shim


WORKSPACE = Path(__file__).resolve().parents[2]
UPSTREAM = WORKSPACE / "third_party" / "E2FGVI"
DEFAULT_CHECKPOINT = (
    WORKSPACE / "research" / "e2fgvi_hq" / "checkpoints" / "E2FGVI-HQ-CVPR22.pth"
)


def load_model(checkpoint: Path) -> torch.nn.Module:
    if not UPSTREAM.is_dir():
        raise FileNotFoundError(f"Official E2FGVI clone not found: {UPSTREAM}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HQ checkpoint not found: {checkpoint}")
    install_mmcv_cpu_shim()
    sys.path.insert(0, str(UPSTREAM))
    from model.e2fgvi_hq import InpaintGenerator

    model = InpaintGenerator(init_weights=False).cpu().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and minimally execute E2FGVI-HQ on CPU")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--forward", action="store_true", help="Run a 2-frame 60x108 forward pass")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    started = time.perf_counter()
    model = load_model(args.checkpoint)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        {
            "model": "E2FGVI-HQ",
            "device": next(model.parameters()).device.type,
            "parameters": parameter_count,
            "load_seconds": round(time.perf_counter() - started, 3),
            "torch": torch.__version__,
            "torchvision_deform_conv": "CPU shim",
            "cuda_available": torch.cuda.is_available(),
        }
    )
    if args.forward:
        sample = torch.zeros((1, 2, 3, 60, 108), dtype=torch.float32)
        with torch.inference_mode():
            forward_started = time.perf_counter()
            prediction, flows = model(sample, num_local_frames=2)
        print(
            {
                "prediction_shape": tuple(prediction.shape),
                "forward_flow_shape": tuple(flows[0].shape),
                "forward_seconds": round(time.perf_counter() - forward_started, 3),
                "finite": bool(torch.isfinite(prediction).all()),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
