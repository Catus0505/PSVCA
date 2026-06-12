from __future__ import annotations

import numpy as np


def _as_2d(X: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must have at least one row")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_1d(y: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def add_intercept(X: np.ndarray) -> np.ndarray:
    arr = _as_2d(X, "X")
    return np.column_stack([np.ones(arr.shape[0], dtype=np.float64), arr])


def ols_predict(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    x_train = _as_2d(X_train, "X_train")
    y_arr = _as_1d(y_train, "y_train")
    x_eval = _as_2d(X_eval, "X_eval")
    if x_train.shape[0] != y_arr.shape[0]:
        raise ValueError("X_train and y_train row counts differ")
    if x_train.shape[1] != x_eval.shape[1]:
        raise ValueError("X_train and X_eval feature counts differ")
    coef, *_ = np.linalg.lstsq(add_intercept(x_train), y_arr, rcond=None)
    return add_intercept(x_eval) @ coef


def residualize(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(Z, dtype=np.float64)
    if z_arr.ndim == 1:
        z_2d = z_arr[:, None]
        squeeze = True
    elif z_arr.ndim == 2:
        z_2d = z_arr
        squeeze = False
    else:
        raise ValueError(f"Z must be 1D or 2D, got shape {z_arr.shape}")
    x_arr = _as_2d(X, "X")
    if z_2d.shape[0] != x_arr.shape[0]:
        raise ValueError("Z and X row counts differ")
    if not np.all(np.isfinite(z_2d)):
        raise ValueError("Z must be finite")
    coef, *_ = np.linalg.lstsq(add_intercept(x_arr), z_2d, rcond=None)
    resid = z_2d - add_intercept(x_arr) @ coef
    return resid[:, 0] if squeeze else resid


def in_sample_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    y_arr = _as_1d(y, "y")
    pred = _as_1d(y_hat, "y_hat")
    if y_arr.shape != pred.shape:
        raise ValueError("y and y_hat shapes differ")
    denom = float(np.sum((y_arr - y_arr.mean()) ** 2))
    if denom <= np.finfo(np.float64).eps:
        raise ValueError("R2 denominator is too small")
    return 1.0 - float(np.sum((y_arr - pred) ** 2)) / denom


def incremental_r2_direct_ols(y: np.ndarray, X_base: np.ndarray, X_add: np.ndarray) -> float:
    y_arr = _as_1d(y, "y")
    base = _as_2d(X_base, "X_base")
    add = _as_2d(X_add, "X_add")
    if base.shape[0] != y_arr.shape[0] or add.shape[0] != y_arr.shape[0]:
        raise ValueError("all inputs must have the same number of rows")
    y_hat_base = ols_predict(base, y_arr, base)
    y_hat_joint = ols_predict(np.column_stack([base, add]), y_arr, np.column_stack([base, add]))
    return in_sample_r2(y_arr, y_hat_joint) - in_sample_r2(y_arr, y_hat_base)


def incremental_r2_fwl_ols(y: np.ndarray, X_base: np.ndarray, X_add: np.ndarray) -> float:
    y_arr = _as_1d(y, "y")
    base = _as_2d(X_base, "X_base")
    add = _as_2d(X_add, "X_add")
    if base.shape[0] != y_arr.shape[0] or add.shape[0] != y_arr.shape[0]:
        raise ValueError("all inputs must have the same number of rows")
    y_resid = residualize(y_arr, base)
    add_resid = residualize(add, base)
    coef, *_ = np.linalg.lstsq(add_resid, y_resid, rcond=None)
    y_add_hat_resid = add_resid @ coef
    denom = float(np.sum((y_arr - y_arr.mean()) ** 2))
    if denom <= np.finfo(np.float64).eps:
        raise ValueError("R2 denominator is too small")
    return float(np.sum(y_add_hat_resid * y_add_hat_resid)) / denom
