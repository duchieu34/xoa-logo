from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    total_memory_bytes: int | None


def resolve_device(requested: str) -> DeviceInfo:
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Select a GPU runtime and install a CUDA-enabled PyTorch build."
            )
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        return DeviceInfo(device, properties.name, properties.total_memory)
    return DeviceInfo(torch.device("cpu"), "CPU", None)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return None
