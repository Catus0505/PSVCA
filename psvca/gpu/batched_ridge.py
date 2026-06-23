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

    train = _as_index_array(train_idx, "train_idx")
    val = _as_index_array(val_idx, "val_idx")
    cert = _as_index_array(cert_idx, "cert_idx")
    n_rows = x_arr.shape[1]
    for name, index in (("train_idx", train), ("val_idx", val), ("cert_idx", cert)):
        if np.any(index < 0) or np.any(index >= n_rows):
            raise ValueError(f"{name} contains out-of-bounds rows")

    torch_dtype = torch.float64 if np.dtype(dtype) == np.dtype(np.float64) else torch.float32
    x = torch.as_tensor(x_arr, dtype=torch_dtype, device=device)
    target = torch.as_tensor(y_arr, dtype=torch_dtype, device=device)
    alphas = torch.as_tensor(alpha_arr, dtype=torch_dtype, device=device)
    train_t = torch.as_tensor(train, dtype=torch.long, device=device)
    val_t = torch.as_tensor(val, dtype=torch.long, device=device)
    cert_t = torch.as_tensor(cert, dtype=torch.long, device=device)

    x_train = x.index_select(1, train_t)
    y_train = target.index_select(0, train_t)
    x_val = x.index_select(1, val_t)
    y_val = target.index_select(0, val_t)
    x_cert = x.index_select(1, cert_t)
    y_cert = target.index_select(0, cert_t)

    x_mean = x_train.mean(dim=1)
    y_mean = y_train.mean()
    x_train_centered = x_train - x_mean[:, None, :]
    y_train_centered = y_train - y_mean

    u, s, vh = torch.linalg.svd(x_train_centered, full_matrices=False)
    uy = torch.matmul(u.transpose(-2, -1), y_train_centered.expand(x.shape[0], -1).unsqueeze(-1)).squeeze(-1)
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
    mse = torch.mean((y_val[None, :, None] - pred_val) ** 2, dim=1)
    alpha_idx = torch.argmin(mse, dim=1)

    batch_idx = torch.arange(x.shape[0], device=device)
    coef = coef_path[batch_idx, alpha_idx]
    intercept = intercept_path[batch_idx, alpha_idx]
    pred_cert = torch.sum(x_cert * coef[:, None, :], dim=2) + intercept[:, None]

    centered = y_cert - y_cert.mean()
    sst = torch.sum(centered * centered)
    sse = torch.sum((y_cert[None, :] - pred_cert) ** 2, dim=1)
    r2 = torch.where(
        sst <= torch.as_tensor(float(variance_eps), dtype=torch_dtype, device=device),
        torch.full_like(sse, float("nan")),
        1.0 - sse / sst,
    )

    alpha_idx_np = _tensor_to_numpy(alpha_idx, dtype=np.int64)
    return BatchedRidgeResult(
        coef=_tensor_to_numpy(coef),
        intercept=_tensor_to_numpy(intercept),
        alpha_idx=alpha_idx_np,
        alpha=alpha_arr[alpha_idx_np],
        r2_cert=_tensor_to_numpy(r2),
        pred_cert=_tensor_to_numpy(pred_cert),
    )
