from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from research.e2fgvi_hq.device import resolve_device


def test_resolve_cpu_device() -> None:
    info = resolve_device("cpu")
    assert info.device == torch.device("cpu")
    assert info.name == "CPU"
    assert info.total_memory_bytes is None


def test_resolve_cuda_fails_clearly_when_unavailable() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("cuda")


def test_resolve_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("mps")
