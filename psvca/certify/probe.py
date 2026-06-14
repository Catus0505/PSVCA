from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from psvca.certify.gates import GateConfig, evaluate_pairwise_gates
from psvca.linalg.svd_ridge import RidgePath, fit_ridge_path
from psvca.nulls.row_perm import row_permute_1d


@dataclass(frozen=True)
class PairwiseProbeConfig:
    alphas: tuple[float, ...]
    B: int
    seed: int
    null_method: str = "phase"
    alpha_rule: str = "val_grid"
    variance_eps: float = 1e-12
    sparse_eps: float = 1e-12


@dataclass(frozen=True)
class PairwiseProbeResult:
    target: int
    source: int
    delta_true: float
    delta_null: np.ndarray
    delta_null_mean: float
    delta_null_std: float
    aligned_gain: float
    p_value: float
    B: int
    alpha_own: float
    alpha_joint: float
    alpha_null: np.ndarray
    gate_delta_true: bool
    gate_aligned_gain: bool
    near_zero_target_variance: bool
    sparse_zero: bool
    unstable_metric: bool
    certified_candidate: bool
    n_train_fit: int
    n_val_alpha: int
    n_cert: int


@dataclass(frozen=True)
class CandidateGroupProbeResult:
    target: int
    source: int
    mode: str
    group_id: str
    group_size: int
    delta_true: float
    delta_null: np.ndarray
    delta_null_mean: float
    delta_null_std: float
    aligned_gain: float
    p_value: float
    B: int
    alpha_reduced: float
    alpha_full: float
    alpha_null: np.ndarray
    alpha_rule: str
    gate_delta_true: bool
    gate_aligned_gain: bool
    near_zero_target_variance: bool
    sparse_zero: bool
    unstable_metric: bool
    certified_candidate: bool
    n_train_fit: int
    n_val_alpha: int
    n_cert: int


def normalize_n_jobs(n_jobs: int) -> int:
    if int(n_jobs) == -1:
        return int(os.cpu_count() or 1)
    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be >=1, or -1 for all available CPUs")
    return int(n_jobs)


def _as_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_2d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must have at least one row")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one column")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _check_rows(y: np.ndarray, X: np.ndarray, y_name: str, x_name: str) -> None:
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"{y_name} and {x_name} row counts differ")


def _fit_select_alpha(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: np.ndarray,
    alpha_rule: str,
) -> tuple[RidgePath, float]:
    path = fit_ridge_path(X_train, y_train, alphas)
    if alpha_rule == "gcv":
        return path, path.best_gcv_alpha()
    if alpha_rule != "val_grid":
        raise ValueError(f"unsupported alpha_rule: {alpha_rule}")
    mse = np.array(
        [np.mean((y_val - path.predict(X_val, float(alpha))) ** 2) for alpha in path.alphas]
    )
    return path, float(path.alphas[int(np.argmin(mse))])


def _delta_from_predictions(
    *,
    y_cert: np.ndarray,
    pred_own: np.ndarray,
    pred_joint: np.ndarray,
    variance_eps: float,
) -> float:
    centered = y_cert - y_cert.mean()
    sst = float(np.sum(centered * centered))
    if sst <= variance_eps:
        return float("nan")
    sse_own = float(np.sum((y_cert - pred_own) ** 2))
    sse_joint = float(np.sum((y_cert - pred_joint) ** 2))
    return (1.0 - sse_joint / sst) - (1.0 - sse_own / sst)


def _delta_reduced_full(
    *,
    y_cert: np.ndarray,
    pred_reduced: np.ndarray,
    pred_full: np.ndarray,
    variance_eps: float,
) -> float:
    return _delta_from_predictions(
        y_cert=y_cert,
        pred_own=pred_reduced,
        pred_joint=pred_full,
        variance_eps=variance_eps,
    )


def _row_permute_design(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_cert: np.ndarray,
    *,
    seed: int,
    surrogate_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged = np.vstack([X_train, X_val, X_cert])
    randomized = np.empty_like(merged)
    for col in range(merged.shape[1]):
        randomized[:, col] = row_permute_1d(
            merged[:, col], seed=seed + 9176 * (surrogate_id + 1) + col
        )
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    return (
        randomized[:n_train],
        randomized[n_train : n_train + n_val],
        randomized[n_train + n_val :],
    )


def _surrogate_blocks(item) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(item, dict):
        return item["train"], item["val"], item["cert"]
    if isinstance(item, (tuple, list)) and len(item) == 3:
        return item[0], item[1], item[2]
    raise TypeError("surrogate_bank entries must be dicts or (train, val, cert) tuples")


def probe_pairwise(
    *,
    target: int,
    source: int,
    y_train,
    y_val,
    y_cert,
    own_train,
    own_val,
    own_cert,
    source_train,
    source_val,
    source_cert,
    surrogate_bank=None,
    config: PairwiseProbeConfig,
) -> PairwiseProbeResult:
    if target == source:
        raise ValueError("target and source must differ for pairwise probe")
    if config.B <= 0:
        raise ValueError("B must be positive")
    alphas = np.asarray(config.alphas, dtype=np.float64)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("config.alphas must be a non-empty 1D sequence")

    y_train_arr = _as_1d(y_train, "y_train")
    y_val_arr = _as_1d(y_val, "y_val")
    y_cert_arr = _as_1d(y_cert, "y_cert")
    own_train_arr = _as_2d(own_train, "own_train")
    own_val_arr = _as_2d(own_val, "own_val")
    own_cert_arr = _as_2d(own_cert, "own_cert")
    source_train_arr = _as_2d(source_train, "source_train")
    source_val_arr = _as_2d(source_val, "source_val")
    source_cert_arr = _as_2d(source_cert, "source_cert")
    for y, X, y_name, x_name in (
        (y_train_arr, own_train_arr, "y_train", "own_train"),
        (y_val_arr, own_val_arr, "y_val", "own_val"),
        (y_cert_arr, own_cert_arr, "y_cert", "own_cert"),
        (y_train_arr, source_train_arr, "y_train", "source_train"),
        (y_val_arr, source_val_arr, "y_val", "source_val"),
        (y_cert_arr, source_cert_arr, "y_cert", "source_cert"),
    ):
        _check_rows(y, X, y_name, x_name)

    own_path, alpha_own = _fit_select_alpha(
        X_train=own_train_arr,
        y_train=y_train_arr,
        X_val=own_val_arr,
        y_val=y_val_arr,
        alphas=alphas,
        alpha_rule=config.alpha_rule,
    )
    pred_own_cert = own_path.predict(own_cert_arr, alpha_own)

    joint_train = np.column_stack([own_train_arr, source_train_arr])
    joint_val = np.column_stack([own_val_arr, source_val_arr])
    joint_cert = np.column_stack([own_cert_arr, source_cert_arr])
    joint_path, alpha_joint = _fit_select_alpha(
        X_train=joint_train,
        y_train=y_train_arr,
        X_val=joint_val,
        y_val=y_val_arr,
        alphas=alphas,
        alpha_rule=config.alpha_rule,
    )
    pred_joint_cert = joint_path.predict(joint_cert, alpha_joint)
    delta_true = _delta_from_predictions(
        y_cert=y_cert_arr,
        pred_own=pred_own_cert,
        pred_joint=pred_joint_cert,
        variance_eps=config.variance_eps,
    )

    delta_null = []
    alpha_null = []
    bank = list(surrogate_bank) if surrogate_bank is not None else None
    if bank is not None and len(bank) != config.B:
        raise ValueError("surrogate_bank length must equal B")
    if bank is None and config.null_method == "phase":
        raise ValueError(
            "phase null for lagged designs requires source-level surrogate_bank"
        )
    for b in range(config.B):
        if bank is not None:
            s_train, s_val, s_cert = _surrogate_blocks(bank[b])
            s_train = _as_2d(s_train, f"surrogate_train_{b}")
            s_val = _as_2d(s_val, f"surrogate_val_{b}")
            s_cert = _as_2d(s_cert, f"surrogate_cert_{b}")
        elif config.null_method == "row_perm":
            s_train, s_val, s_cert = _row_permute_design(
                source_train_arr,
                source_val_arr,
                source_cert_arr,
                seed=config.seed,
                surrogate_id=b,
            )
        else:
            raise ValueError(f"unsupported null_method: {config.null_method}")

        null_train = np.column_stack([own_train_arr, s_train])
        null_val = np.column_stack([own_val_arr, s_val])
        null_cert = np.column_stack([own_cert_arr, s_cert])
        null_path, alpha_b = _fit_select_alpha(
            X_train=null_train,
            y_train=y_train_arr,
            X_val=null_val,
            y_val=y_val_arr,
            alphas=alphas,
            alpha_rule=config.alpha_rule,
        )
        pred_null_cert = null_path.predict(null_cert, alpha_b)
        delta_b = _delta_from_predictions(
            y_cert=y_cert_arr,
            pred_own=pred_own_cert,
            pred_joint=pred_null_cert,
            variance_eps=config.variance_eps,
        )
        delta_null.append(delta_b)
        alpha_null.append(alpha_b)

    delta_null_arr = np.asarray(delta_null, dtype=np.float64)
    alpha_null_arr = np.asarray(alpha_null, dtype=np.float64)
    if np.isfinite(delta_true) and np.all(np.isfinite(delta_null_arr)):
        delta_null_mean = float(delta_null_arr.mean())
        delta_null_std = float(delta_null_arr.std())
        aligned_gain = float(delta_true - delta_null_mean)
        p_value = float((1 + np.count_nonzero(delta_null_arr >= delta_true)) / (config.B + 1))
    else:
        delta_null_mean = float("nan")
        delta_null_std = float("nan")
        aligned_gain = float("nan")
        p_value = float("nan")

    gates = evaluate_pairwise_gates(
        delta_true=delta_true,
        aligned_gain=aligned_gain,
        y_cert=y_cert_arr,
        source_design_cert=source_cert_arr,
        gate_config=GateConfig(
            variance_eps=config.variance_eps,
            sparse_eps=config.sparse_eps,
        ),
    )
    return PairwiseProbeResult(
        target=int(target),
        source=int(source),
        delta_true=float(delta_true),
        delta_null=delta_null_arr,
        delta_null_mean=delta_null_mean,
        delta_null_std=delta_null_std,
        aligned_gain=float(aligned_gain),
        p_value=p_value,
        B=int(config.B),
        alpha_own=float(alpha_own),
        alpha_joint=float(alpha_joint),
        alpha_null=alpha_null_arr,
        gate_delta_true=gates.gate_delta_true,
        gate_aligned_gain=gates.gate_aligned_gain,
        near_zero_target_variance=gates.near_zero_target_variance,
        sparse_zero=gates.sparse_zero,
        unstable_metric=gates.unstable_metric,
        certified_candidate=gates.certified_candidate,
        n_train_fit=int(len(y_train_arr)),
        n_val_alpha=int(len(y_val_arr)),
        n_cert=int(len(y_cert_arr)),
    )


def probe_candidate_group(
    *,
    target: int,
    source: int,
    group_sources: tuple[int, ...],
    y_train,
    y_val,
    y_cert,
    own_train,
    own_val,
    own_cert,
    source_train_by_source: dict[int, np.ndarray],
    source_val_by_source: dict[int, np.ndarray],
    source_cert_by_source: dict[int, np.ndarray],
    surrogate_bank=None,
    group_id: str | None = None,
    n_jobs: int = 1,
    config: PairwiseProbeConfig,
) -> CandidateGroupProbeResult:
    normalize_n_jobs(n_jobs)
    if source == target:
        raise ValueError("target and source must differ for candidate-group probe")
    if source not in group_sources:
        raise ValueError("source must be a member of group_sources")
    if target in group_sources:
        raise ValueError("group_sources must not contain target")
    if len(set(group_sources)) != len(group_sources):
        raise ValueError("group_sources must not contain duplicates")
    if config.B <= 0:
        raise ValueError("B must be positive")
    alphas = np.asarray(config.alphas, dtype=np.float64)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("config.alphas must be a non-empty 1D sequence")

    y_train_arr = _as_1d(y_train, "y_train")
    y_val_arr = _as_1d(y_val, "y_val")
    y_cert_arr = _as_1d(y_cert, "y_cert")
    own_train_arr = _as_2d(own_train, "own_train")
    own_val_arr = _as_2d(own_val, "own_val")
    own_cert_arr = _as_2d(own_cert, "own_cert")
    for y, X, y_name, x_name in (
        (y_train_arr, own_train_arr, "y_train", "own_train"),
        (y_val_arr, own_val_arr, "y_val", "own_val"),
        (y_cert_arr, own_cert_arr, "y_cert", "own_cert"),
    ):
        _check_rows(y, X, y_name, x_name)

    train_blocks: dict[int, np.ndarray] = {}
    val_blocks: dict[int, np.ndarray] = {}
    cert_blocks: dict[int, np.ndarray] = {}
    for group_source in group_sources:
        train_blocks[group_source] = _as_2d(
            source_train_by_source[group_source], f"source_train_{group_source}"
        )
        val_blocks[group_source] = _as_2d(
            source_val_by_source[group_source], f"source_val_{group_source}"
        )
        cert_blocks[group_source] = _as_2d(
            source_cert_by_source[group_source], f"source_cert_{group_source}"
        )
        _check_rows(y_train_arr, train_blocks[group_source], "y_train", f"source_train_{group_source}")
        _check_rows(y_val_arr, val_blocks[group_source], "y_val", f"source_val_{group_source}")
        _check_rows(y_cert_arr, cert_blocks[group_source], "y_cert", f"source_cert_{group_source}")

    other_sources = tuple(s for s in group_sources if s != source)

    def stack_design(own: np.ndarray, blocks: dict[int, np.ndarray], sources: tuple[int, ...]) -> np.ndarray:
        if not sources:
            return own
        return np.column_stack([own, *(blocks[s] for s in sources)])

    reduced_train = stack_design(own_train_arr, train_blocks, other_sources)
    reduced_val = stack_design(own_val_arr, val_blocks, other_sources)
    reduced_cert = stack_design(own_cert_arr, cert_blocks, other_sources)
    full_train = stack_design(own_train_arr, train_blocks, group_sources)
    full_val = stack_design(own_val_arr, val_blocks, group_sources)
    full_cert = stack_design(own_cert_arr, cert_blocks, group_sources)

    reduced_path, alpha_reduced = _fit_select_alpha(
        X_train=reduced_train,
        y_train=y_train_arr,
        X_val=reduced_val,
        y_val=y_val_arr,
        alphas=alphas,
        alpha_rule=config.alpha_rule,
    )
    pred_reduced_cert = reduced_path.predict(reduced_cert, alpha_reduced)
    full_path, alpha_full = _fit_select_alpha(
        X_train=full_train,
        y_train=y_train_arr,
        X_val=full_val,
        y_val=y_val_arr,
        alphas=alphas,
        alpha_rule=config.alpha_rule,
    )
    pred_full_cert = full_path.predict(full_cert, alpha_full)
    delta_true = _delta_reduced_full(
        y_cert=y_cert_arr,
        pred_reduced=pred_reduced_cert,
        pred_full=pred_full_cert,
        variance_eps=config.variance_eps,
    )

    delta_null = []
    alpha_null = []
    bank = list(surrogate_bank) if surrogate_bank is not None else None
    if bank is not None and len(bank) != config.B:
        raise ValueError("surrogate_bank length must equal B")
    if bank is None and config.null_method == "phase":
        raise ValueError(
            "phase null for lagged designs requires source-level surrogate_bank"
        )
    for b in range(config.B):
        if bank is not None:
            s_train, s_val, s_cert = _surrogate_blocks(bank[b])
            s_train = _as_2d(s_train, f"surrogate_train_{b}")
            s_val = _as_2d(s_val, f"surrogate_val_{b}")
            s_cert = _as_2d(s_cert, f"surrogate_cert_{b}")
        elif config.null_method == "row_perm":
            s_train, s_val, s_cert = _row_permute_design(
                train_blocks[source],
                val_blocks[source],
                cert_blocks[source],
                seed=config.seed,
                surrogate_id=b,
            )
        else:
            raise ValueError(f"unsupported null_method: {config.null_method}")
        _check_rows(y_train_arr, s_train, "y_train", f"surrogate_train_{b}")
        _check_rows(y_val_arr, s_val, "y_val", f"surrogate_val_{b}")
        _check_rows(y_cert_arr, s_cert, "y_cert", f"surrogate_cert_{b}")

        null_train_blocks = dict(train_blocks)
        null_val_blocks = dict(val_blocks)
        null_cert_blocks = dict(cert_blocks)
        null_train_blocks[source] = s_train
        null_val_blocks[source] = s_val
        null_cert_blocks[source] = s_cert
        null_train = stack_design(own_train_arr, null_train_blocks, group_sources)
        null_val = stack_design(own_val_arr, null_val_blocks, group_sources)
        null_cert = stack_design(own_cert_arr, null_cert_blocks, group_sources)
        null_path, alpha_b = _fit_select_alpha(
            X_train=null_train,
            y_train=y_train_arr,
            X_val=null_val,
            y_val=y_val_arr,
            alphas=alphas,
            alpha_rule=config.alpha_rule,
        )
        pred_null_cert = null_path.predict(null_cert, alpha_b)
        delta_b = _delta_reduced_full(
            y_cert=y_cert_arr,
            pred_reduced=pred_reduced_cert,
            pred_full=pred_null_cert,
            variance_eps=config.variance_eps,
        )
        delta_null.append(delta_b)
        alpha_null.append(alpha_b)

    delta_null_arr = np.asarray(delta_null, dtype=np.float64)
    alpha_null_arr = np.asarray(alpha_null, dtype=np.float64)
    if np.isfinite(delta_true) and np.all(np.isfinite(delta_null_arr)):
        delta_null_mean = float(delta_null_arr.mean())
        delta_null_std = float(delta_null_arr.std())
        aligned_gain = float(delta_true - delta_null_mean)
        p_value = float((1 + np.count_nonzero(delta_null_arr >= delta_true)) / (config.B + 1))
    else:
        delta_null_mean = float("nan")
        delta_null_std = float("nan")
        aligned_gain = float("nan")
        p_value = float("nan")

    gates = evaluate_pairwise_gates(
        delta_true=delta_true,
        aligned_gain=aligned_gain,
        y_cert=y_cert_arr,
        source_design_cert=cert_blocks[source],
        gate_config=GateConfig(
            variance_eps=config.variance_eps,
            sparse_eps=config.sparse_eps,
        ),
    )
    return CandidateGroupProbeResult(
        target=int(target),
        source=int(source),
        mode="candidate_group",
        group_id=group_id or f"target_{int(target)}_top{len(group_sources)}",
        group_size=int(len(group_sources)),
        delta_true=float(delta_true),
        delta_null=delta_null_arr,
        delta_null_mean=delta_null_mean,
        delta_null_std=delta_null_std,
        aligned_gain=float(aligned_gain),
        p_value=p_value,
        B=int(config.B),
        alpha_reduced=float(alpha_reduced),
        alpha_full=float(alpha_full),
        alpha_null=alpha_null_arr,
        alpha_rule=config.alpha_rule,
        gate_delta_true=gates.gate_delta_true,
        gate_aligned_gain=gates.gate_aligned_gain,
        near_zero_target_variance=gates.near_zero_target_variance,
        sparse_zero=gates.sparse_zero,
        unstable_metric=gates.unstable_metric,
        certified_candidate=gates.certified_candidate,
        n_train_fit=int(len(y_train_arr)),
        n_val_alpha=int(len(y_val_arr)),
        n_cert=int(len(y_cert_arr)),
    )
