from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from psvca.base.innovation import compute_innovation_for_target
from psvca.linalg.design import make_lagged_design
from psvca.linalg.svd_ridge import fit_ridge_path


@dataclass(frozen=True)
class ValueScreenConfig:
    top_m: int = 4
    max_targets: int | None = None
    targets: tuple[int, ...] | None = None
    seed: int = 0
    alpha_rule: str = "val_grid"


@dataclass(frozen=True)
class ValueScreenResult:
    edges: pd.DataFrame
    summary: dict


def _bounds(split) -> tuple[int, int]:
    if not hasattr(split, "start") or not hasattr(split, "end"):
        raise TypeError("split objects must expose start and end attributes")
    return int(split.start), int(split.end)


def _source_design(values: np.ndarray, target: int, source: int, split, lookback: int, horizon: int):
    start, end = _bounds(split)
    return make_lagged_design(
        values,
        target=target,
        sources=(source,),
        lookback=lookback,
        horizon=horizon,
        y_start=start,
        y_end=end,
        include_own=False,
    )


def _select_targets(n_channels: int, cfg: ValueScreenConfig) -> tuple[int, ...]:
    if cfg.targets is not None:
        targets = tuple(int(t) for t in cfg.targets)
    else:
        limit = n_channels if cfg.max_targets is None else min(int(cfg.max_targets), n_channels)
        targets = tuple(range(limit))
    if not targets:
        raise ValueError("at least one target is required")
    for target in targets:
        if target < 0 or target >= n_channels:
            raise ValueError(f"target index out of bounds: {target}")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must not contain duplicates")
    return targets


def _fit_alpha(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: np.ndarray,
    alpha_rule: str,
) -> tuple[object, float]:
    path = fit_ridge_path(X_train, y_train, alphas)
    if alpha_rule == "gcv":
        return path, path.best_gcv_alpha()
    if alpha_rule != "val_grid":
        raise ValueError(f"unsupported alpha_rule: {alpha_rule}")
    mse = np.array(
        [np.mean((y_val - path.predict(X_val, float(alpha))) ** 2) for alpha in path.alphas],
        dtype=np.float64,
    )
    return path, float(path.alphas[int(np.argmin(mse))])


def _residual_oos_r2(y_true: np.ndarray, pred: np.ndarray, *, variance_eps: float = 1e-12) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(pred, dtype=np.float64)
    if y.ndim != 1 or y_hat.ndim != 1 or y.shape != y_hat.shape:
        raise ValueError("y_true and pred must be 1D arrays with the same shape")
    if y.size == 0:
        raise ValueError("y_true must not be empty")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(y_hat)):
        return float("nan")
    sst = float(np.sum(y * y))
    if sst <= variance_eps:
        return float("nan")
    sse = float(np.sum((y - y_hat) ** 2))
    return 1.0 - sse / sst


def run_value_screen(
    *,
    values: np.ndarray,
    channels: tuple[str, ...],
    splits,
    lookback: int,
    horizon: int,
    alphas,
    config: ValueScreenConfig | None = None,
) -> ValueScreenResult:
    cfg = config or ValueScreenConfig()
    if cfg.top_m <= 0:
        raise ValueError("top_m must be positive")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"values must have shape (T, N), got {arr.shape}")
    if len(channels) != arr.shape[1]:
        raise ValueError("channels length must match values columns")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    alpha_arr = np.asarray(alphas, dtype=np.float64)
    if alpha_arr.ndim != 1 or alpha_arr.size == 0:
        raise ValueError("alphas must be a non-empty 1D sequence")

    n_channels = arr.shape[1]
    targets = _select_targets(n_channels, cfg)
    rows: list[dict] = []

    for target in targets:
        innovation = compute_innovation_for_target(
            values=arr,
            channels=channels,
            splits=splits,
            target=target,
            lookback=lookback,
            horizon=horizon,
            alphas=alpha_arr,
            seed=cfg.seed,
        )
        target_rows: list[dict] = []
        for source in range(n_channels):
            if source == target:
                continue
            source_train = _source_design(arr, target, source, splits.train_fit, lookback, horizon)
            source_val = _source_design(arr, target, source, splits.val_alpha, lookback, horizon)
            source_cert = _source_design(arr, target, source, splits.cert, lookback, horizon)
            path, alpha = _fit_alpha(
                X_train=source_train.X,
                y_train=innovation.resid_train_fit,
                X_val=source_val.X,
                y_val=innovation.resid_val_alpha,
                alphas=alpha_arr,
                alpha_rule=cfg.alpha_rule,
            )
            pred_cert = path.predict(source_cert.X, alpha)
            score = _residual_oos_r2(innovation.resid_cert, pred_cert)
            target_rows.append(
                {
                    "target": int(target),
                    "source": int(source),
                    "s_screen": float(score),
                    "screen_rank": 0,
                    "passed_screen": False,
                    "n_train_fit": int(len(innovation.resid_train_fit)),
                    "n_val_alpha": int(len(innovation.resid_val_alpha)),
                    "n_cert": int(len(innovation.resid_cert)),
                    "alpha_screen": float(alpha),
                    "alpha_rule": cfg.alpha_rule,
                }
            )

        target_rows.sort(
            key=lambda row: (
                not np.isfinite(row["s_screen"]),
                -row["s_screen"] if np.isfinite(row["s_screen"]) else np.inf,
                row["source"],
            )
        )
        for rank, row in enumerate(target_rows, start=1):
            row["screen_rank"] = int(rank)
            row["passed_screen"] = bool(rank <= cfg.top_m and np.isfinite(row["s_screen"]))
            rows.append(row)

    edges = pd.DataFrame(rows)
    summary = {
        "n_targets_screened": int(len(targets)),
        "top_m": int(cfg.top_m),
        "n_screen_edges": int(edges["passed_screen"].sum()) if not edges.empty else 0,
        "alpha_rule": cfg.alpha_rule,
    }
    return ValueScreenResult(edges=edges, summary=summary)
