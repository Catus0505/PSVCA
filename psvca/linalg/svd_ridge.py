from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgePath:
    alphas: np.ndarray
    coef_path: np.ndarray
    intercepts: np.ndarray
    x_mean: np.ndarray
    y_mean: float
    singular_values: np.ndarray
    df: np.ndarray
    gcv_scores: np.ndarray

    def _alpha_index(self, alpha: float) -> int:
        matches = np.flatnonzero(self.alphas == alpha)
        if len(matches) != 1:
            raise ValueError(f"alpha {alpha!r} is not in this ridge path")
        return int(matches[0])

    def predict(self, X: np.ndarray, alpha: float) -> np.ndarray:
        return self.predict_index(X, self._alpha_index(alpha))

    def predict_index(self, X: np.ndarray, index: int) -> np.ndarray:
        arr = _validate_X_eval(X, self.x_mean.shape[0])
        if index < 0 or index >= len(self.alphas):
            raise IndexError(f"alpha index out of bounds: {index}")
        return arr @ self.coef_path[index] + self.intercepts[index]

    def best_gcv_alpha(self) -> float:
        if len(self.gcv_scores) == 0:
            raise ValueError("empty GCV score path")
        return float(self.alphas[int(np.argmin(self.gcv_scores))])


def _validate_X_y(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {x_arr.shape}")
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if x_arr.shape[0] == 0 or x_arr.shape[1] == 0:
        raise ValueError("X must have at least one row and one column")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("X and y must be finite")
    return x_arr, y_arr


def _validate_alphas(alphas: np.ndarray) -> np.ndarray:
    arr = np.asarray(alphas, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("alphas must be 1D")
    if arr.size == 0:
        raise ValueError("alphas must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("alphas must be finite")
    if np.any(arr < 0):
        raise ValueError("alphas must be non-negative")
    if len(np.unique(arr)) != arr.size:
        raise ValueError("alphas must not contain duplicates")
    return arr


def _validate_X_eval(X: np.ndarray, n_features: int) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {arr.shape}")
    if arr.shape[1] != n_features:
        raise ValueError(f"X has {arr.shape[1]} features, expected {n_features}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X must be finite")
    return arr


def fit_ridge_path(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> RidgePath:
    x_arr, y_arr = _validate_X_y(X, y)
    alpha_arr = _validate_alphas(alphas)

    x_mean = x_arr.mean(axis=0)
    y_mean = float(y_arr.mean())
    x_centered = x_arr - x_mean
    y_centered = y_arr - y_mean
    u, s, vh = np.linalg.svd(x_centered, full_matrices=False)
    uy = u.T @ y_centered

    coef_path = []
    intercepts = []
    dfs = []
    gcv_scores = []
    n = x_arr.shape[0]
    s2 = s * s
    pinv_cutoff = np.finfo(np.float64).eps * max(x_arr.shape) * (s[0] if s.size else 0.0)
    for alpha in alpha_arr:
        if alpha == 0.0:
            shrink = np.divide(1.0, s, out=np.zeros_like(s), where=s > pinv_cutoff)
        else:
            shrink = np.divide(s, s2 + alpha, out=np.zeros_like(s), where=(s2 + alpha) != 0)
        coef = vh.T @ (shrink * uy)
        intercept = y_mean - float(x_mean @ coef)
        y_hat_centered = x_centered @ coef
        residual = y_centered - y_hat_centered
        mse = float(np.mean(residual * residual))
        # df is the centered ridge smoother trace, sum s^2/(s^2+alpha).
        # The unpenalized intercept is handled by centering and is not added here.
        if alpha == 0.0:
            df = float(np.count_nonzero(s > pinv_cutoff))
        else:
            df = float(np.sum(np.divide(s2, s2 + alpha, out=np.zeros_like(s2), where=(s2 + alpha) != 0)))
        denom = 1.0 - df / n
        if denom <= np.finfo(np.float64).eps:
            gcv = np.inf
        else:
            gcv = mse / (denom * denom)
        coef_path.append(coef)
        intercepts.append(intercept)
        dfs.append(df)
        gcv_scores.append(gcv)

    return RidgePath(
        alphas=alpha_arr.copy(),
        coef_path=np.vstack(coef_path),
        intercepts=np.asarray(intercepts, dtype=np.float64),
        x_mean=x_mean,
        y_mean=y_mean,
        singular_values=s,
        df=np.asarray(dfs, dtype=np.float64),
        gcv_scores=np.asarray(gcv_scores, dtype=np.float64),
    )


def fit_ridge_single(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    path = fit_ridge_path(X, y, np.asarray([alpha], dtype=np.float64))
    return path.coef_path[0].copy(), float(path.intercepts[0])


def r2_score_oos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    baseline: float | None = None,
) -> float:
    y_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_arr.ndim != 1 or pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D")
    if y_arr.shape != pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_arr.size == 0:
        raise ValueError("y_true must not be empty")
    if not np.all(np.isfinite(y_arr)) or not np.all(np.isfinite(pred_arr)):
        raise ValueError("y_true and y_pred must be finite")
    base = float(y_arr.mean() if baseline is None else baseline)
    denom = float(np.sum((y_arr - base) ** 2))
    if denom <= np.finfo(np.float64).eps:
        raise ValueError("R2 denominator is too small")
    num = float(np.sum((y_arr - pred_arr) ** 2))
    return 1.0 - num / denom
