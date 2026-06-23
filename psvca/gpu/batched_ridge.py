from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from psvca.gpu.device import resolve_gpu_device


@dataclass(frozen=True)
class BatchedRidgeResult:
    coef: np.ndarray
    intercept: np.ndarray
    alpha_idx: np.ndarray
    alpha: np.ndarray
    r2_cert: np.ndarray
    pred_cert: np.ndarray


@dataclass(frozen=True)
class BatchedRidgeFit:
    coef: np.ndarray
    intercept: np.ndarray
    alpha_idx: np.ndarray
    alpha: np.ndarray


def _as_index_array(index, name: str) -> np.ndarray:
    arr = np.asarray(index, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _tensor_to_numpy(tensor, dtype=None) -> np.ndarray:
    cpu_tensor = tensor.detach().cpu()
    try:
        arr = cpu_tensor.numpy()
    except RuntimeError:
        arr = np.asarray(cpu_tensor.tolist())
    if dtype is not None:
        return arr.astype(dtype, copy=False)
    return np.asarray(arr)


def batched_ridge_svd(
    X: np.ndarray,
    y: np.ndarray,
    alpha_grid,
    *,
    train_idx,
    val_idx,
    cert_idx,
    dtype: str | np.dtype = "float32",
    device: str | None = None,
    variance_eps: float = 1e-12,
) -> BatchedRidgeResult:
    """Fit a batch of centered ridge models with one SVD per batch item.

    Alpha selection matches psvca.certify.probe._fit_select_alpha for
    alpha_rule="val_grid": every alpha is scored on validation MSE and the
    per-item argmin is selected. Returned predictions and R2 are on cert_idx.
    """
    import torch

    device = resolve_gpu_device(device=device)

    x_arr = np.asarray(X)
    y_arr = np.asarray(y)
    alpha_arr = np.asarray(alpha_grid, dtype=np.float64)
    if x_arr.ndim != 3:
        raise ValueError(f"X must have shape (batch, rows, cols), got {x_arr.shape}")
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
    if x_arr.shape[1] != y_arr.shape[0]:
        raise ValueError("X rows must match y length")
    if alpha_arr.ndim != 1 or alpha_arr.size == 0:
        raise ValueError("alpha_grid must be a non-empty 1D sequence")
    if np.any(alpha_arr < 0) or not np.all(np.isfinite(alpha_arr)):
        raise ValueError("alpha_grid must contain finite non-negative values")
    if np.any(alpha_arr < 0) or not np.all(np.isfinite(alpha_arr)):
        raise ValueError("alpha_grid must contain finite non-negative values")

    train = _as_index_array(train_idx, "train_idx")
    val = _as_index_array(val_idx, "val_idx")
    cert = _as_index_array(cert_idx, "cert_idx")
    n_rows = x_arr.shape[1]
    for name, index in (("train_idx", train), ("val_idx", val), ("cert_idx", cert)):
        if np.any(index < 0) or np.any(index >= n_rows):
            raise ValueError(f"{name} contains out-of-bounds rows")

    fit = batched_ridge_fit(
        X_train=x_arr[:, train],
        y_train=y_arr[train],
        X_val=x_arr[:, val],
        y_val=y_arr[val],
        alpha_grid=alpha_arr,
        dtype=dtype,
        device=device,
    )
    pred_cert = predict_batched(fit=fit, X=x_arr[:, cert])
    r2 = r2_cert_score(y_arr[cert], pred_cert, variance_eps=variance_eps)
    return BatchedRidgeResult(
        coef=fit.coef,
        intercept=fit.intercept,
        alpha_idx=fit.alpha_idx,
        alpha=fit.alpha,
        r2_cert=r2,
        pred_cert=pred_cert,
    )


def batched_ridge_fit(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha_grid,
    dtype: str | np.dtype = "float32",
    device: str | None = None,
) -> BatchedRidgeFit:
    import torch

    device = resolve_gpu_device(device=device)
    train_arr = np.asarray(X_train)
    val_arr = np.asarray(X_val)
    y_train_arr = np.asarray(y_train)
    y_val_arr = np.asarray(y_val)
    alpha_arr = np.asarray(alpha_grid, dtype=np.float64)
    if train_arr.ndim != 3 or val_arr.ndim != 3:
        raise ValueError("X_train and X_val must have shape (batch, rows, cols)")
    if train_arr.shape[0] != val_arr.shape[0] or train_arr.shape[2] != val_arr.shape[2]:
        raise ValueError("X_train and X_val batch/column dimensions must match")
    if train_arr.shape[1] != y_train_arr.shape[0] or val_arr.shape[1] != y_val_arr.shape[0]:
        raise ValueError("X/y row counts differ")
    if alpha_arr.ndim != 1 or alpha_arr.size == 0:
        raise ValueError("alpha_grid must be a non-empty 1D sequence")

    torch_dtype = torch.float64 if np.dtype(dtype) == np.dtype(np.float64) else torch.float32
    x_train = torch.as_tensor(train_arr, dtype=torch_dtype, device=device)
    x_val = torch.as_tensor(val_arr, dtype=torch_dtype, device=device)
    y_train_t = torch.as_tensor(y_train_arr, dtype=torch_dtype, device=device)
    y_val_t = torch.as_tensor(y_val_arr, dtype=torch_dtype, device=device)
    alphas = torch.as_tensor(alpha_arr, dtype=torch_dtype, device=device)

    x_mean = x_train.mean(dim=1)
    y_mean = y_train_t.mean()
    x_train_centered = x_train - x_mean[:, None, :]
    y_train_centered = y_train_t - y_mean

    u, s, vh = torch.linalg.svd(x_train_centered, full_matrices=False)
    uy = torch.matmul(
        u.transpose(-2, -1),
        y_train_centered.expand(x_train.shape[0], -1).unsqueeze(-1),
    ).squeeze(-1)
    s2 = s * s

    eps = torch.finfo(torch_dtype).eps
    max_shape = max(int(x_train.shape[1]), int(x_train.shape[2]))
    cutoff = eps * max_shape * s[:, 0]
    shrink = []
    for alpha in torch.unbind(alphas):
        if float(alpha.detach().cpu()) == 0.0:
            shrink_alpha = torch.zeros_like(s)
            mask = s > cutoff[:, None]
            shrink_alpha[mask] = 1.0 / s[mask]
        else:
            denom = s2 + alpha
            shrink_alpha = torch.where(denom != 0, s / denom, torch.zeros_like(s))
        shrink.append(shrink_alpha)
    shrink_t = torch.stack(shrink, dim=1)

    weighted = shrink_t * uy[:, None, :]
    coef_path = torch.matmul(vh.transpose(-2, -1)[:, None, :, :], weighted.unsqueeze(-1)).squeeze(-1)
    intercept_path = y_mean - torch.sum(x_mean[:, None, :] * coef_path, dim=2)
    pred_val = torch.einsum("brc,bac->bra", x_val, coef_path) + intercept_path[:, None, :]
    mse = torch.mean((y_val_t[None, :, None] - pred_val) ** 2, dim=1)
    alpha_idx = torch.argmin(mse, dim=1)

    batch_idx = torch.arange(x_train.shape[0], device=device)
    alpha_idx_np = _tensor_to_numpy(alpha_idx, dtype=np.int64)
    return BatchedRidgeFit(
        coef=_tensor_to_numpy(coef_path[batch_idx, alpha_idx]),
        intercept=_tensor_to_numpy(intercept_path[batch_idx, alpha_idx]),
        alpha_idx=alpha_idx_np,
        alpha=alpha_arr[alpha_idx_np],
    )


def predict_batched(*, fit: BatchedRidgeFit, X: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(X)
    if x_arr.ndim != 3:
        raise ValueError("X must have shape (batch, rows, cols)")
    return np.einsum("brc,bc->br", x_arr, fit.coef) + fit.intercept[:, None]


def r2_cert_score(y_cert: np.ndarray, pred_cert: np.ndarray, *, variance_eps: float) -> np.ndarray:
    y_arr = np.asarray(y_cert)
    pred_arr = np.asarray(pred_cert)
    centered = y_arr - y_arr.mean()
    sst = float(np.sum(centered * centered))
    if sst <= float(variance_eps):
        return np.full(pred_arr.shape[0], float("nan"), dtype=np.float64)
    sse = np.sum((y_arr[None, :] - pred_arr) ** 2, axis=1)
    return 1.0 - sse / sst
