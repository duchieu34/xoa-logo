from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


LAMA_MODEL_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"


@dataclass(frozen=True)
class CropBox:
    x: int
    y: int
    width: int
    height: int


def context_crop_box(
    frame_width: int,
    frame_height: int,
    logo_bbox: tuple[int, int, int, int],
    size: int,
) -> CropBox:
    if size <= 0 or size % 8:
        raise ValueError("LaMa context size must be positive and divisible by 8")
    if size > frame_width or size > frame_height:
        raise ValueError("LaMa context crop cannot exceed frame dimensions")
    logo_x, logo_y, logo_width, logo_height = logo_bbox
    center_x = logo_x + logo_width / 2
    center_y = logo_y + logo_height / 2
    x = int(np.clip(round(center_x - size / 2), 0, frame_width - size))
    y = int(np.clip(round(center_y - size / 2), 0, frame_height - size))
    if not (
        x <= logo_x
        and y <= logo_y
        and x + size >= logo_x + logo_width
        and y + size >= logo_y + logo_height
    ):
        raise ValueError("Computed context crop does not contain the full logo")
    return CropBox(x=x, y=y, width=size, height=size)


def mask_in_context(
    roi_mask: np.ndarray,
    roi_px: tuple[int, int, int, int],
    crop: CropBox,
) -> np.ndarray:
    roi_x, roi_y, roi_width, roi_height = roi_px
    if roi_mask.shape != (roi_height, roi_width):
        raise ValueError("ROI mask dimensions do not match ROI")
    output = np.zeros((crop.height, crop.width), dtype=np.uint8)
    destination_x = roi_x - crop.x
    destination_y = roi_y - crop.y
    x1, y1 = max(0, destination_x), max(0, destination_y)
    x2 = min(crop.width, destination_x + roi_width)
    y2 = min(crop.height, destination_y + roi_height)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("ROI does not overlap LaMa context crop")
    source_x = x1 - destination_x
    source_y = y1 - destination_y
    output[y1:y2, x1:x2] = roi_mask[
        source_y:source_y + (y2 - y1), source_x:source_x + (x2 - x1)
    ]
    return output


def composite_mask_only(original: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if original.shape != prediction.shape or original.shape[:2] != mask.shape:
        raise ValueError("Original, prediction and mask dimensions must match")
    output = original.copy()
    selected = mask > 0
    output[selected] = prediction[selected]
    return output


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream checksum used for artifact identity
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LamaCpu:
    def __init__(self, model_path: Path, threads: int | None = None) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"LaMa model not found: {model_path}. Run scripts/download_lama.py first."
            )
        actual_md5 = _md5(model_path)
        if actual_md5 != LAMA_MODEL_MD5:
            raise ValueError(
                f"Unexpected LaMa model checksum: expected {LAMA_MODEL_MD5}, got {actual_md5}"
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch CPU is required; install requirements-ai.txt") from exc
        if threads is not None:
            torch.set_num_threads(max(1, threads))
        self.torch = torch
        self.model = torch.jit.load(str(model_path), map_location="cpu").eval()
        self.model_path = model_path
        self.model_md5 = actual_md5

    def predict(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if bgr.ndim != 3 or bgr.shape[2] != 3 or mask.shape != bgr.shape[:2]:
            raise ValueError("LaMa expects BGR HxWx3 image and matching HxW mask")
        if bgr.shape[0] % 8 or bgr.shape[1] % 8:
            raise ValueError("LaMa input dimensions must be divisible by 8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_array = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        mask_array = np.ascontiguousarray((mask > 0)[None], dtype=np.float32)
        image_tensor = self.torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = self.torch.from_numpy(mask_array).unsqueeze(0)
        with self.torch.inference_mode():
            prediction = self.model(image_tensor, mask_tensor)
        rgb_result = prediction[0].permute(1, 2, 0).detach().cpu().numpy()
        rgb_result = np.clip(rgb_result * 255.0, 0, 255).astype(np.uint8)
        predicted_bgr = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2BGR)
        return composite_mask_only(bgr, predicted_bgr, mask)

    @property
    def runtime_info(self) -> dict[str, object]:
        return {
            "torch_version": self.torch.__version__,
            "torch_cuda_build": self.torch.version.cuda,
            "cuda_available": bool(self.torch.cuda.is_available()),
            "device": "cpu",
            "threads": int(self.torch.get_num_threads()),
        }
