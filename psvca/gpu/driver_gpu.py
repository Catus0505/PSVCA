from __future__ import annotations

from dataclasses import asdict
import os

import numpy as np
import pandas as pd

from psvca.certify.gates import GateConfig, evaluate_pairwise_gates
from psvca.certify.probe import CandidateGroupProbeResult
from psvca.gpu.batched_ridge import batched_ridge_svd


def run_gpu_batch(
    *,
    driver,
    target_groups: dict[int, tuple[int, ...]],
    metadata_for=None,
) -> pd.DataFrame:
    """Run candidate-group certification with batched GPU ridge/SVD.

    Phase P1 keeps lagged-design and surrogate construction on the existing
    numpy path, then batches same-shaped reduced/full/null ridge fits per
    target on cuda:0.
    """
    normalized_groups = {
        int(target): tuple(int(s) for s in group_sources if int(s) != int(target))
        for target, group_sources in target_groups.items()
    }
    normalized_groups = {
        target: group_sources
        for target, group_sources in normalized_groups.items()
        if group_sources
    }
    rows: list[dict] = []
    for target, group_sources in normalized_groups.items():
        rows.extend(_target_rows(driver, target, group_sources, metadata_for=metadata_for))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target", "source"]).reset_index(drop=True)


def _target_rows(driver, target: int, group_sources: tuple[int, ...], *, metadata_for=None) -> list[dict]:
    own_train, own_val, own_cert = driver.own_designs(target)
    train_blocks, val_blocks, cert_blocks = driver.source_design_dicts(group_sources)
    y = np.concatenate([own_train.y, own_val.y, own_cert.y])
    train_idx, val_idx, cert_idx = _split_indices(
        len(own_train.y),
        len(own_val.y),
        len(own_cert.y),
    )
    cfg = driver.probe_config
    if cfg.alpha_rule != "val_grid":
        raise ValueError(f"GPU batched ridge supports alpha_rule='val_grid', got {cfg.alpha_rule!r}")
    if cfg.null_method != "phase":
        raise ValueError(f"GPU candidate-group backend supports null_method='phase', got {cfg.null_method!r}")
    alphas = np.asarray(cfg.alphas, dtype=np.float64)
    gpu_dtype = _gpu_dtype(driver)
    gpu_device = _gpu_device(driver)
    group_id = f"target_{int(target)}_top{len(group_sources)}"

    reduced_designs = []
    full_designs = []
    for source in group_sources:
        other_sources = tuple(s for s in group_sources if s != source)
        reduced_designs.append(
            _merge_splits(
                _stack_design(own_train.X, train_blocks, other_sources),
                _stack_design(own_val.X, val_blocks, other_sources),
                _stack_design(own_cert.X, cert_blocks, other_sources),
            )
        )
        full_designs.append(
            _merge_splits(
                _stack_design(own_train.X, train_blocks, group_sources),
                _stack_design(own_val.X, val_blocks, group_sources),
                _stack_design(own_cert.X, cert_blocks, group_sources),
            )
        )

    reduced = batched_ridge_svd(
        np.stack(reduced_designs, axis=0),
        y,
        alphas,
        train_idx=train_idx,
        val_idx=val_idx,
        cert_idx=cert_idx,
        dtype=gpu_dtype,
        device=gpu_device,
        variance_eps=cfg.variance_eps,
    )
    full = batched_ridge_svd(
        np.stack(full_designs, axis=0),
        y,
        alphas,
        train_idx=train_idx,
        val_idx=val_idx,
        cert_idx=cert_idx,
        dtype=gpu_dtype,
        device=gpu_device,
        variance_eps=cfg.variance_eps,
    )

    delta_true = full.r2_cert - reduced.r2_cert
    survivors = [
        i
        for i, delta in enumerate(delta_true)
        if not (cfg.skip_null_on_fail and float(delta) <= float(cfg.delta_floor))
    ]
    null_by_edge: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if survivors:
        null_designs = []
        null_edge_indices = []
        for edge_index in survivors:
            source = int(group_sources[edge_index])
            for s_train, s_val, s_cert in driver.surrogate_bank(target=target, source=source):
                null_train_blocks = dict(train_blocks)
                null_val_blocks = dict(val_blocks)
                null_cert_blocks = dict(cert_blocks)
                null_train_blocks[source] = s_train
                null_val_blocks[source] = s_val
                null_cert_blocks[source] = s_cert
                null_designs.append(
                    _merge_splits(
                        _stack_design(own_train.X, null_train_blocks, group_sources),
                        _stack_design(own_val.X, null_val_blocks, group_sources),
                        _stack_design(own_cert.X, null_cert_blocks, group_sources),
                    )
                )
                null_edge_indices.append(edge_index)
        null = batched_ridge_svd(
            np.stack(null_designs, axis=0),
            y,
            alphas,
            train_idx=train_idx,
            val_idx=val_idx,
            cert_idx=cert_idx,
            dtype=gpu_dtype,
            device=gpu_device,
            variance_eps=cfg.variance_eps,
        )
        for edge_index in survivors:
            mask = np.asarray(null_edge_indices, dtype=np.int64) == int(edge_index)
            null_by_edge[int(edge_index)] = (
                null.r2_cert[mask] - reduced.r2_cert[int(edge_index)],
                null.alpha[mask],
            )

    rows = []
    for edge_index, source in enumerate(group_sources):
        delta_null, alpha_null = null_by_edge.get(
            int(edge_index),
            (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)),
        )
        skipped_null = bool(cfg.skip_null_on_fail and float(delta_true[edge_index]) <= float(cfg.delta_floor))
        if skipped_null:
            delta_null_mean = float("nan")
            delta_null_std = float("nan")
            aligned_gain = float("nan")
            p_value = 1.0
        elif np.isfinite(delta_true[edge_index]) and np.all(np.isfinite(delta_null)):
            delta_null_mean = float(delta_null.mean())
            delta_null_std = float(delta_null.std())
            aligned_gain = float(delta_true[edge_index] - delta_null_mean)
            p_value = float((1 + np.count_nonzero(delta_null >= delta_true[edge_index])) / (cfg.B + 1))
        else:
            delta_null_mean = float("nan")
            delta_null_std = float("nan")
            aligned_gain = float("nan")
            p_value = float("nan")

        gates = evaluate_pairwise_gates(
            delta_true=float(delta_true[edge_index]),
            aligned_gain=aligned_gain,
            y_cert=own_cert.y,
            source_design_cert=cert_blocks[int(source)],
            gate_config=GateConfig(
                variance_eps=cfg.variance_eps,
                sparse_eps=cfg.sparse_eps,
            ),
        )
        result = CandidateGroupProbeResult(
            target=int(target),
            source=int(source),
            mode="candidate_group",
            group_id=group_id,
            group_size=int(len(group_sources)),
            delta_true=float(delta_true[edge_index]),
            delta_null=np.asarray(delta_null, dtype=np.float64),
            delta_null_mean=delta_null_mean,
            delta_null_std=delta_null_std,
            aligned_gain=aligned_gain,
            p_value=p_value,
            B=int(cfg.B),
            alpha_reduced=float(reduced.alpha[edge_index]),
            alpha_full=float(full.alpha[edge_index]),
            alpha_null=np.asarray(alpha_null, dtype=np.float64),
            alpha_rule=cfg.alpha_rule,
            gate_delta_true=gates.gate_delta_true,
            gate_aligned_gain=gates.gate_aligned_gain,
            near_zero_target_variance=gates.near_zero_target_variance,
            sparse_zero=gates.sparse_zero,
            unstable_metric=gates.unstable_metric,
            certified_candidate=gates.certified_candidate,
            skipped_null=skipped_null,
            n_train_fit=int(len(own_train.y)),
            n_val_alpha=int(len(own_val.y)),
            n_cert=int(len(own_cert.y)),
        )
        row = _result_row(result)
        metadata = metadata_for(int(target), int(source)) if metadata_for else None
        if metadata:
            row.update(metadata)
        rows.append(row)
    return rows


def _split_indices(n_train: int, n_val: int, n_cert: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.arange(0, int(n_train), dtype=np.int64)
    val = np.arange(int(n_train), int(n_train + n_val), dtype=np.int64)
    cert = np.arange(int(n_train + n_val), int(n_train + n_val + n_cert), dtype=np.int64)
    return train, val, cert


def _gpu_dtype(driver) -> str:
    raw = str(getattr(driver, "gpu_dtype", os.environ.get("PSVCA_GPU_DTYPE", "float32"))).lower()
    if raw in {"fp64", "float64", "double"}:
        return "float64"
    if raw in {"fp32", "float32", "single"}:
        return "float32"
    raise ValueError(f"unsupported GPU dtype: {raw!r}")


def _gpu_device(driver) -> str:
    return str(getattr(driver, "gpu_device", os.environ.get("PSVCA_GPU_DEVICE", "cuda:0")))


def _merge_splits(train: np.ndarray, val: np.ndarray, cert: np.ndarray) -> np.ndarray:
    return np.vstack([train, val, cert])


def _stack_design(own: np.ndarray, blocks: dict[int, np.ndarray], sources: tuple[int, ...]) -> np.ndarray:
    if not sources:
        return own
    return np.column_stack([own, *(blocks[int(source)] for source in sources)])


def _result_row(result: CandidateGroupProbeResult) -> dict:
    data = asdict(result)
    data.pop("delta_null", None)
    data.pop("alpha_null", None)
    return data
