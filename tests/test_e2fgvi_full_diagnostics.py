from __future__ import annotations

import numpy as np

from research.e2fgvi_hq.render_full_validation_diagnostics import _tight_box


def test_tight_box_adds_margin_and_clamps_to_image() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[15:19, 25:29] = 255
    assert _tight_box(mask, margin=3) == (22, 12, 8, 8)
