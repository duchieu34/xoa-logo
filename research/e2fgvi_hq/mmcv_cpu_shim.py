from __future__ import annotations

import math
import sys
import types
from typing import Any

import torch
from torch import nn
from torchvision.ops import deform_conv2d as torchvision_deform_conv2d


class ConvModule(nn.Module):
    """Subset of mmcv.cnn.ConvModule used by E2FGVI's SPyNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        norm_cfg: dict[str, Any] | None = None,
        act_cfg: dict[str, Any] | None = dict(type="ReLU"),
        **_: Any,
    ) -> None:
        super().__init__()
        if norm_cfg is not None:
            raise NotImplementedError("The E2FGVI CPU shim only supports norm_cfg=None")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if act_cfg is None:
            self.activate: nn.Module | None = None
        elif act_cfg.get("type") == "ReLU":
            self.activate = nn.ReLU(inplace=bool(act_cfg.get("inplace", True)))
        else:
            raise NotImplementedError(f"Unsupported activation: {act_cfg}")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv(value)
        return self.activate(value) if self.activate is not None else value


class ModulatedDeformConv2d(nn.Module):
    """Parameter-compatible mmcv deformable conv backed by torchvision CPU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, value: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return modulated_deform_conv2d(
            value,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )


def modulated_deform_conv2d(
    value: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
    groups: int,
    deform_groups: int,
) -> torch.Tensor:
    if groups != 1:
        raise NotImplementedError("Grouped deformable convolution is not needed by E2FGVI-HQ")
    expected_offset_channels = 2 * deform_groups * weight.shape[-2] * weight.shape[-1]
    expected_mask_channels = deform_groups * weight.shape[-2] * weight.shape[-1]
    if offset.shape[1] != expected_offset_channels or mask.shape[1] != expected_mask_channels:
        raise ValueError(
            "E2FGVI deformable-convolution tensor shape mismatch: "
            f"offset={offset.shape[1]}/{expected_offset_channels}, "
            f"mask={mask.shape[1]}/{expected_mask_channels}"
        )
    return torchvision_deform_conv2d(
        value,
        offset,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=mask,
    )


def constant_init(module: nn.Module, val: float, bias: float = 0.0) -> None:
    if getattr(module, "weight", None) is not None:
        nn.init.constant_(module.weight, val)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


def load_checkpoint(*_: Any, **__: Any) -> None:
    """Skip SPyNet's URL fetch; the official HQ state dict supplies these weights."""


def install_mmcv_cpu_shim() -> None:
    """Expose only the mmcv symbols imported by untouched upstream E2FGVI code."""
    if "mmcv" in sys.modules:
        return
    mmcv = types.ModuleType("mmcv")
    cnn = types.ModuleType("mmcv.cnn")
    ops = types.ModuleType("mmcv.ops")
    runner = types.ModuleType("mmcv.runner")
    cnn.ConvModule = ConvModule
    cnn.constant_init = constant_init
    ops.ModulatedDeformConv2d = ModulatedDeformConv2d
    ops.modulated_deform_conv2d = modulated_deform_conv2d
    runner.load_checkpoint = load_checkpoint
    mmcv.cnn = cnn
    mmcv.ops = ops
    mmcv.runner = runner
    sys.modules.update(
        {"mmcv": mmcv, "mmcv.cnn": cnn, "mmcv.ops": ops, "mmcv.runner": runner}
    )
