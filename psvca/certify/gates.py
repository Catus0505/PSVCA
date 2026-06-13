from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateConfig:
    variance_eps: float = 1e-12
    sparse_eps: float = 1e-12
    min_effective_rows: int = 8


@dataclass(frozen=True)
class GateResult:
    gate_delta_true: bool
    gate_aligned_gain: bool
    near_zero_target_variance: bool
    sparse_zero: bool
    unstable_metric: bool
    certified_candidate: bool


def _as_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _as_2d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def evaluate_pairwise_gates(
    *,
    delta_true: float,
    aligned_gain: float,
    y_cert,
    source_design_cert,
    gate_config: GateConfig | None = None,
) -> GateResult:
    cfg = gate_config or GateConfig()
    y = _as_1d(y_cert, "y_cert")
    source = _as_2d(source_design_cert, "source_design_cert")
    if source.shape[0] != y.shape[0]:
        raise ValueError("source_design_cert and y_cert row counts differ")

    finite_metrics = np.isfinite(delta_true) and np.isfinite(aligned_gain)
    finite_inputs = np.all(np.isfinite(y)) and np.all(np.isfinite(source))
    centered_y = y - y.mean() if len(y) else y
    sst = float(np.sum(centered_y * centered_y)) if len(y) else 0.0
    near_zero_target_variance = (
        len(y) < cfg.min_effective_rows
        or not np.all(np.isfinite(y))
        or float(np.var(y)) <= cfg.variance_eps
        or sst <= cfg.variance_eps
    )
    sparse_zero = (
        source.shape[0] < cfg.min_effective_rows
        or source.shape[1] == 0
        or not np.all(np.isfinite(source))
        or float(np.max(np.abs(source))) <= cfg.sparse_eps
        or float(np.linalg.norm(source)) <= cfg.sparse_eps
    )
    gate_delta_true = bool(delta_true > 0.0) if np.isfinite(delta_true) else False
    gate_aligned_gain = bool(aligned_gain > 0.0) if np.isfinite(aligned_gain) else False
    unstable_metric = bool(
        (not finite_metrics)
        or (not finite_inputs)
        or near_zero_target_variance
        or sparse_zero
    )
    certified_candidate = bool(
        gate_delta_true and gate_aligned_gain and not unstable_metric
    )
    return GateResult(
        gate_delta_true=gate_delta_true,
        gate_aligned_gain=gate_aligned_gain,
        near_zero_target_variance=bool(near_zero_target_variance),
        sparse_zero=bool(sparse_zero),
        unstable_metric=unstable_metric,
        certified_candidate=certified_candidate,
    )
