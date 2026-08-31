from __future__ import annotations

from veo_watermark_remover.mask_calibration import _sample_indices


def test_sample_indices_cover_whole_video() -> None:
    indices = _sample_indices(192)
    assert len(indices) == 8
    assert indices[0] == 0
    assert indices[-1] == 191


def test_sample_indices_handle_short_video() -> None:
    assert _sample_indices(3) == [0, 1, 2]
