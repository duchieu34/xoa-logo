from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import RelativeROI
from .video_io import VideoMetadata


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write diagnostic image: {path}")


def draw_roi(frame: np.ndarray, roi_px: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi_px
    result = frame.copy()
    thickness = max(2, round(min(frame.shape[:2]) / 500))
    cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), (0, 255, 255), thickness)
    cv2.putText(
        result, "candidate ROI", (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, min(frame.shape[:2]) / 1400), (0, 255, 255), thickness, cv2.LINE_AA,
    )
    return result


def make_contact_sheet(images: list[np.ndarray], labels: list[str], columns: int = 4) -> np.ndarray:
    if not images:
        raise ValueError("At least one image is required")
    target_width = 420
    tiles: list[np.ndarray] = []
    for image, label in zip(images, labels, strict=True):
        scale = target_width / image.shape[1]
        resized = cv2.resize(image, (target_width, round(image.shape[0] * scale)))
        canvas = cv2.copyMakeBorder(resized, 34, 8, 8, 8, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(canvas)
    tile_height = max(tile.shape[0] for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    blank = np.full((tile_height, target_width + 16, 3), 24, dtype=np.uint8)
    padded = [cv2.copyMakeBorder(tile, 0, tile_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)) for tile in tiles]
    padded.extend(blank.copy() for _ in range(rows * columns - len(padded)))
    return np.vstack([np.hstack(padded[row * columns:(row + 1) * columns]) for row in range(rows)])


def save_report(
    path: Path,
    metadata: VideoMetadata,
    roi: RelativeROI,
    roi_px: tuple[int, int, int, int],
    frame_records: list[dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    payload = {
        "experiment": 0,
        "removal_performed": False,
        "video": asdict(metadata),
        "candidate_roi_relative": asdict(roi),
        "candidate_roi_pixels": dict(zip(("x", "y", "width", "height"), roi_px, strict=True)),
        "frames": frame_records,
        "elapsed_seconds": elapsed_seconds,
        "warning": "Candidate ROI is not a calibrated watermark mask.",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

