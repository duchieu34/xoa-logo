from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from research.e2fgvi_hq.benchmark import _center_overlap_weight, _pad_for_hq
from research.e2fgvi_hq.mmcv_cpu_shim import ConvModule, modulated_deform_conv2d


def test_conv_module_keeps_mmcv_checkpoint_key_layout() -> None:
    module = ConvModule(3, 4, 3, padding=1, norm_cfg=None, act_cfg=None)
    assert set(module.state_dict()) == {"conv.weight", "conv.bias"}


def test_cpu_deform_conv_matches_regular_conv_at_zero_offset() -> None:
    value = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    weight = torch.ones((1, 1, 3, 3), dtype=torch.float32)
    offset = torch.zeros((1, 18, 5, 5), dtype=torch.float32)
    mask = torch.ones((1, 9, 5, 5), dtype=torch.float32)
    actual = modulated_deform_conv2d(
        value, offset, mask, weight, None, 1, 1, 1, groups=1, deform_groups=1
    )
    expected = functional.conv2d(value, weight, padding=1)
    torch.testing.assert_close(actual, expected)


def test_hq_mirror_padding_for_256_crop() -> None:
    value = torch.zeros((1, 2, 3, 256, 256), dtype=torch.float32)
    padded, original = _pad_for_hq(value)
    assert original == (256, 256)
    assert padded.shape == (1, 2, 3, 300, 324)


def test_hq_mirror_padding_candidate_crop_shapes() -> None:
    expected = {192: (240, 216), 224: (240, 324), 256: (300, 324)}
    for size, padded_size in expected.items():
        value = torch.zeros((1, 2, 3, size, size), dtype=torch.float32)
        padded, original = _pad_for_hq(value)
        assert original == (size, size)
        assert padded.shape[-2:] == padded_size


def test_center_overlap_weight_prefers_window_center_symmetrically() -> None:
    weights = [
        _center_overlap_weight(index, center=5, neighbor_stride=5)
        for index in range(11)
    ]
    assert weights == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]


def test_center_overlap_weight_rejects_frame_outside_window() -> None:
    with pytest.raises(ValueError, match="outside"):
        _center_overlap_weight(frame_index=11, center=5, neighbor_stride=5)
