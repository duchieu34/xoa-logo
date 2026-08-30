from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelativeROI:
    """A resolution-independent rectangle, expressed in frame-relative units."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("ROI values must be between 0 and 1")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("ROI must fit inside the frame")

    @classmethod
    def parse(cls, raw: str) -> "RelativeROI":
        try:
            values = [float(value.strip()) for value in raw.split(",")]
        except ValueError as exc:
            raise ValueError("ROI must contain four numbers: x,y,width,height") from exc
        if len(values) != 4:
            raise ValueError("ROI must contain four numbers: x,y,width,height")
        return cls(*values)

    def to_pixels(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        x1 = round(self.x * frame_width)
        y1 = round(self.y * frame_height)
        x2 = round((self.x + self.width) * frame_width)
        y2 = round((self.y + self.height) * frame_height)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


# A deliberately generous bottom-right candidate area. It must be calibrated from
# the real sample before it can be considered the Veo watermark ROI.
DEFAULT_CANDIDATE_ROI = RelativeROI(0.78, 0.78, 0.21, 0.20)

