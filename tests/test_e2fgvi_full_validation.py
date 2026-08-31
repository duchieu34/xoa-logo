from __future__ import annotations

import numpy as np

from research.e2fgvi_hq.full_video_validation import (
    _crop_boundary_max_difference,
    _sequence_metrics,
)


def test_crop_boundary_metric_ignores_interior_change() -> None:
    original = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    restored = original.copy()
    restored[:, 5:11, 5:11] = 255
    assert _crop_boundary_max_difference(original, restored) == 0


def test_full_sequence_metrics_rank_largest_transition() -> None:
    crops = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    crops[2:] = 100
    mask = np.full((8, 8), 255, dtype=np.uint8)
    metrics = _sequence_metrics(crops, mask)
    assert metrics["mean_temporal_mad"] == round(100 / 3, 6)
    assert metrics["worst_transitions"][0] == {
        "from_frame": 1,
        "to_frame": 2,
        "mad": 100.0,
        "mean_luma_delta": 100.0,
    }
