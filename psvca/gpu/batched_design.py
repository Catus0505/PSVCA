from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from psvca.gpu.device import resolve_gpu_device


@dataclass(frozen=True)
class BatchedDesignMatrix:
    X: np.ndarray
    y: np.ndarray
    future_indices: np.ndarray
    origin_indices: np.ndarray


def batched_lagged_design(
    values: np.ndarray,
    target,
    sources,
    lookback: int,
    horizon: int,
    y_start: int,
    y_end: int,
    *,
    include_own: bool = True,
    dtype: str | np.dtype = "float32",
    device: str | None = None,
    assert_cpu_equiv: bool = False,
) -> BatchedDesignMatrix:
    """Construct lagged designs for a batch of series with torch indexing."""
    import torch

    device = resolve_gpu_device(device=device)

    arr = np.asarray(values)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"values must have shape (batch, T, N), got {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError("values must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if y_start < 0 or y_end < 0 or y_end <= y_start:
        raise ValueError("y_start/y_end must define a non-empty non-negative range")
    if y_end > arr.shape[1]:
        raise ValueError(f"y_end exceeds values length: {y_end} > {arr.shape[1]}")

    batch = int(arr.shape[0])
    target_arr = _normalize_targets(target, batch, arr.shape[2])
    source_rows = _normalize_sources(sources, batch, arr.shape[2])
    channel_rows = []
    for batch_index, source_row in enumerate(source_rows):
        if target_arr[batch_index] in source_row:
            raise ValueError("sources must not contain target")
        channels = []
        if include_own:
            channels.append(int(target_arr[batch_index]))
        channels.extend(int(source) for source in source_row)
        if not channels:
            raise ValueError("at least one feature block is required")
        channel_rows.append(tuple(channels))
    n_channels_per_design = {len(row) for row in channel_rows}
    if len(n_channels_per_design) != 1:
        raise ValueError("all batch items must produce the same number of feature columns")

    future = np.arange(int(y_start), int(y_end), dtype=np.int64)
    origin = future - int(horizon) + 1
    valid = origin - int(lookback) >= 0
    future = future[valid]
    origin = origin[valid]
    if future.size == 0:
        raise ValueError("no valid design rows; check lookback, horizon, and y range")
    window = (origin[:, None] - int(lookback)) + np.arange(int(lookback), dtype=np.int64)[None, :]

    torch_dtype = torch.float64 if np.dtype(dtype) == np.dtype(np.float64) else torch.float32
    x = torch.as_tensor(arr, dtype=torch_dtype, device=device)
    time_idx = torch.as_tensor(window, dtype=torch.long, device=device)
    channel_idx = torch.as_tensor(np.asarray(channel_rows, dtype=np.int64), dtype=torch.long, device=device)
    target_idx = torch.as_tensor(target_arr, dtype=torch.long, device=device)
    future_idx = torch.as_tensor(future, dtype=torch.long, device=device)

    by_time = x.index_select(1, time_idx.reshape(-1)).reshape(
        batch,
        int(future.size),
        int(lookback),
        int(arr.shape[2]),
    )
    gather_idx = channel_idx[:, None, None, :].expand(
        batch,
        int(future.size),
        int(lookback),
        int(channel_idx.shape[1]),
    )
    gathered = torch.gather(by_time, dim=3, index=gather_idx)
    X = gathered.permute(0, 1, 3, 2).reshape(batch, int(future.size), -1)
    batch_idx = torch.arange(batch, dtype=torch.long, device=device)[:, None]
    y = x[batch_idx, future_idx[None, :], target_idx[:, None]]

    result = BatchedDesignMatrix(
        X=_tensor_to_numpy(X),
        y=_tensor_to_numpy(y),
        future_indices=future,
        origin_indices=origin,
    )
    if assert_cpu_equiv:
        _assert_cpu_equiv(
            result,
            arr,
            target_arr,
            source_rows,
            lookback,
            horizon,
            y_start,
            y_end,
            include_own=include_own,
        )
    return result


def _normalize_targets(target, batch: int, n_channels: int) -> np.ndarray:
    arr = np.asarray(target, dtype=np.int64)
    if arr.ndim == 0:
        arr = np.full(batch, int(arr), dtype=np.int64)
    if arr.ndim != 1 or arr.size != batch:
        raise ValueError("target must be a scalar or a 1D array matching batch")
    if np.any(arr < 0) or np.any(arr >= n_channels):
        raise ValueError("target index out of bounds")
    return arr


def _normalize_sources(sources, batch: int, n_channels: int) -> list[tuple[int, ...]]:
    if _is_source_tuple(sources):
        rows = [tuple(int(source) for source in sources)] * batch
    elif isinstance(sources, Sequence) and len(sources) == batch:
        rows = [tuple(int(source) for source in row) for row in sources]
    else:
        raise ValueError("sources must be a tuple of channels or a sequence matching batch")
    for row in rows:
        if len(set(row)) != len(row):
            raise ValueError("sources must not contain duplicates")
        if any(source < 0 or source >= n_channels for source in row):
            raise ValueError("source index out of bounds")
    return rows


def _is_source_tuple(value) -> bool:
    return isinstance(value, tuple) and all(isinstance(item, (int, np.integer)) for item in value)


def _tensor_to_numpy(tensor) -> np.ndarray:
    cpu_tensor = tensor.detach().cpu()
    try:
        return np.asarray(cpu_tensor.numpy())
    except RuntimeError:
        return np.asarray(cpu_tensor.tolist())


def _assert_cpu_equiv(
    result: BatchedDesignMatrix,
    values: np.ndarray,
    target: np.ndarray,
    sources: list[tuple[int, ...]],
    lookback: int,
    horizon: int,
    y_start: int,
    y_end: int,
    *,
    include_own: bool,
) -> None:
    from psvca.linalg.design import make_lagged_design

    for batch_index in range(values.shape[0]):
        expected = make_lagged_design(
            values[batch_index],
            int(target[batch_index]),
            sources[batch_index],
            lookback,
            horizon,
            y_start,
            y_end,
            include_own=include_own,
        )
        np.testing.assert_array_equal(result.future_indices, expected.future_indices)
        np.testing.assert_array_equal(result.origin_indices, expected.origin_indices)
        np.testing.assert_allclose(result.X[batch_index], expected.X, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result.y[batch_index], expected.y, rtol=0.0, atol=0.0)
