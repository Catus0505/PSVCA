from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DesignMatrix:
    X: np.ndarray
    y: np.ndarray
    future_indices: np.ndarray
    origin_indices: np.ndarray
    feature_names: tuple[str, ...]


def _validate_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"values must have shape (T, N), got {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("values must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    return arr


def _validate_channel(index: int, n_channels: int, name: str) -> None:
    if index < 0 or index >= n_channels:
        raise ValueError(f"{name} index out of bounds: {index}")


def _window(values: np.ndarray, channel: int, start: int, end: int) -> np.ndarray:
    return values[start:end, channel]


def make_lagged_design(
    values: np.ndarray,
    target: int,
    sources: tuple[int, ...],
    lookback: int,
    horizon: int,
    y_start: int,
    y_end: int,
    *,
    include_own: bool = True,
) -> DesignMatrix:
    arr = _validate_values(values)
    t_total, n_channels = arr.shape
    _validate_channel(target, n_channels, "target")
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if y_start < 0 or y_end < 0 or y_end <= y_start:
        raise ValueError("y_start/y_end must define a non-empty non-negative range")
    if y_end > t_total:
        raise ValueError(f"y_end exceeds values length: {y_end} > {t_total}")
    if len(set(sources)) != len(sources):
        raise ValueError("sources must not contain duplicates")
    if target in sources:
        raise ValueError("sources must not contain target")
    for source in sources:
        _validate_channel(source, n_channels, "source")
    if not include_own and not sources:
        raise ValueError("at least one feature block is required")

    channels: list[tuple[str, int]] = []
    if include_own:
        channels.append(("own", target))
    channels.extend((f"source_{source}", source) for source in sources)

    rows: list[np.ndarray] = []
    y_values: list[float] = []
    future_indices: list[int] = []
    origin_indices: list[int] = []
    for u in range(y_start, y_end):
        origin = u - horizon + 1
        window_start = origin - lookback
        if window_start < 0:
            continue
        if origin > t_total:
            raise ValueError("computed origin exceeds values length")
        row_blocks = [_window(arr, channel, window_start, origin) for _, channel in channels]
        rows.append(np.concatenate(row_blocks))
        y_values.append(float(arr[u, target]))
        future_indices.append(u)
        origin_indices.append(origin)

    if not rows:
        raise ValueError("no valid design rows; check lookback, horizon, and y range")

    feature_names: list[str] = []
    for label, channel in channels:
        for lag_pos in range(lookback):
            lag = lookback - lag_pos
            feature_names.append(f"{label}[{channel}]_lag_{lag}")

    return DesignMatrix(
        X=np.vstack(rows),
        y=np.asarray(y_values, dtype=np.float64),
        future_indices=np.asarray(future_indices, dtype=np.int64),
        origin_indices=np.asarray(origin_indices, dtype=np.int64),
        feature_names=tuple(feature_names),
    )


def make_own_design(
    values: np.ndarray,
    target: int,
    lookback: int,
    horizon: int,
    y_start: int,
    y_end: int,
) -> DesignMatrix:
    return make_lagged_design(
        values,
        target=target,
        sources=(),
        lookback=lookback,
        horizon=horizon,
        y_start=y_start,
        y_end=y_end,
        include_own=True,
    )


def make_source_block(
    values: np.ndarray,
    target: int,
    sources: tuple[int, ...],
    lookback: int,
    horizon: int,
    y_start: int,
    y_end: int,
) -> DesignMatrix:
    return make_lagged_design(
        values,
        target=target,
        sources=sources,
        lookback=lookback,
        horizon=horizon,
        y_start=y_start,
        y_end=y_end,
        include_own=False,
    )
