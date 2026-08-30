from __future__ import annotations

import json
from pathlib import Path


DEFAULT_MEASUREMENT_PATH = Path(__file__).resolve().parent.parent / "assets" / "veo_1080p_measurement.json"


def load_measurement(path: Path = DEFAULT_MEASUREMENT_PATH) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Veo measurement not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def scaled_logo_bbox(
    frame_width: int,
    frame_height: int,
    measurement: dict[str, object],
) -> tuple[int, int, int, int]:
    bbox = measurement["visible_logo_bbox_estimate"]["relative"]  # type: ignore[index]
    x1 = round(float(bbox["x"]) * frame_width)  # type: ignore[index]
    y1 = round(float(bbox["y"]) * frame_height)  # type: ignore[index]
    x2 = round((float(bbox["x"]) + float(bbox["width"])) * frame_width)  # type: ignore[index]
    y2 = round((float(bbox["y"]) + float(bbox["height"])) * frame_height)  # type: ignore[index]
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

