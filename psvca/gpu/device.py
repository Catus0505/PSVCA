from __future__ import annotations

import os


DEFAULT_GPU_DEVICE = "cuda:0"


def resolve_gpu_device(*, driver=None, device: str | None = None) -> str:
    if device is not None:
        return str(device)
    if driver is not None:
        driver_device = getattr(driver, "gpu_device", None)
        if driver_device is not None:
            return str(driver_device)
    return str(os.environ.get("PSVCA_GPU_DEVICE", DEFAULT_GPU_DEVICE))
