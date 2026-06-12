from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from psvca.base.ridge_own import RidgeOwnBase
from psvca.linalg.design import DesignMatrix, make_lagged_design


@dataclass(frozen=True)
class InnovationResult:
    target: int
    channel: str
    alpha_own: float
    train_fit_indices: np.ndarray
    val_alpha_indices: np.ndarray
    cert_indices: np.ndarray
    y_train_fit: np.ndarray
    y_val_alpha: np.ndarray
    y_cert: np.ndarray
    pred_train_fit: np.ndarray
    pred_val_alpha: np.ndarray
    pred_cert: np.ndarray
    resid_train_fit: np.ndarray
    resid_val_alpha: np.ndarray
    resid_cert: np.ndarray


def _range_bounds(split) -> tuple[int, int]:
    if not hasattr(split, "start") or not hasattr(split, "end"):
        raise TypeError("split objects must expose start and end attributes")
    return int(split.start), int(split.end)


def _own_design_for_split(
    values: np.ndarray,
    target: int,
    lookback: int,
    horizon: int,
    split,
) -> DesignMatrix:
    start, end = _range_bounds(split)
    return make_lagged_design(
        values,
        target=target,
        sources=(),
        lookback=lookback,
        horizon=horizon,
        y_start=start,
        y_end=end,
        include_own=True,
    )


def _assert_indices_within(name: str, indices: np.ndarray, split) -> None:
    start, end = _range_bounds(split)
    if len(indices) == 0:
        raise ValueError(f"{name} has no valid future indices")
    if np.any(indices < start) or np.any(indices >= end):
        raise ValueError(f"{name} future indices are outside split range")


def compute_innovation_for_target(
    values: np.ndarray,
    channels: tuple[str, ...],
    splits,
    target: int,
    lookback: int,
    horizon: int,
    alphas: np.ndarray,
    seed: int = 0,
) -> InnovationResult:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"values must have shape (T, N), got {arr.shape}")
    if target < 0 or target >= arr.shape[1]:
        raise ValueError(f"target index out of bounds: {target}")
    if len(channels) != arr.shape[1]:
        raise ValueError("channels length must match values columns")
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")

    train_design = _own_design_for_split(
        arr, target, lookback, horizon, splits.train_fit
    )
    val_design = _own_design_for_split(arr, target, lookback, horizon, splits.val_alpha)
    cert_design = _own_design_for_split(arr, target, lookback, horizon, splits.cert)

    _assert_indices_within("train_fit", train_design.future_indices, splits.train_fit)
    _assert_indices_within("val_alpha", val_design.future_indices, splits.val_alpha)
    _assert_indices_within("cert", cert_design.future_indices, splits.cert)
    if train_design.future_indices.max(initial=-1) >= splits.original_test.start:
        raise ValueError("train_fit design reaches original test split")
    if val_design.future_indices.max(initial=-1) >= splits.original_test.start:
        raise ValueError("val_alpha design reaches original test split")
    if cert_design.future_indices.max(initial=-1) >= splits.original_test.start:
        raise ValueError("cert design reaches original test split")

    model = RidgeOwnBase(alphas=np.asarray(alphas, dtype=np.float64), alpha_rule="gcv")
    model.fit(train_design.X, train_design.y, seed=seed)
    pred_train = model.predict(train_design.X)
    pred_val = model.predict(val_design.X)
    pred_cert = model.predict(cert_design.X)

    return InnovationResult(
        target=target,
        channel=str(channels[target]),
        alpha_own=model.alpha_,
        train_fit_indices=train_design.future_indices,
        val_alpha_indices=val_design.future_indices,
        cert_indices=cert_design.future_indices,
        y_train_fit=train_design.y,
        y_val_alpha=val_design.y,
        y_cert=cert_design.y,
        pred_train_fit=pred_train,
        pred_val_alpha=pred_val,
        pred_cert=pred_cert,
        resid_train_fit=train_design.y - pred_train,
        resid_val_alpha=val_design.y - pred_val,
        resid_cert=cert_design.y - pred_cert,
    )


def compute_innovations_for_targets(
    values: np.ndarray,
    channels: tuple[str, ...],
    splits,
    targets: tuple[int, ...],
    lookback: int,
    horizon: int,
    alphas: np.ndarray,
    seed: int = 0,
) -> tuple[InnovationResult, ...]:
    return tuple(
        compute_innovation_for_target(
            values=values,
            channels=channels,
            splits=splits,
            target=target,
            lookback=lookback,
            horizon=horizon,
            alphas=alphas,
            seed=seed,
        )
        for target in targets
    )
