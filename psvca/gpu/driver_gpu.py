from __future__ import annotations

from dataclasses import asdict
import os

import numpy as np
import pandas as pd

from psvca.certify.gates import GateConfig, evaluate_pairwise_gates
from psvca.certify.probe import CandidateGroupProbeResult
from psvca.gpu.batched_design import batched_lagged_design
from psvca.gpu.device import resolve_gpu_device
from psvca.gpu.batched_ridge import (
    BatchedRidgeFit,
    BatchedRidgeResult,
    batched_ridge_fit,
    batched_ridge_svd,
    predict_batched,
    r2_cert_score,
)
from psvca.gpu.batched_surrogate import batched_phase_surrogate_pairs


def run_gpu_batch(
    *,
    driver,
    target_groups: dict[int, tuple[int, ...]],
    metadata_for=None,
    cert_splits=None,
):
    """Run candidate-group certification with batched GPU design/null/ridge."""
    normalized_groups = {
        int(target): tuple(int(s) for s in group_sources if int(s) != int(target))
        for target, group_sources in target_groups.items()
    }
    normalized_groups = {
        target: group_sources
        for target, group_sources in normalized_groups.items()
        if group_sources
    }
    if cert_splits is not None:
        block_rows: list[list[dict]] = [[] for _ in cert_splits]
        for target, group_sources in normalized_groups.items():
            target_blocks = _target_block_rows(
                driver,
                target,
                group_sources,
                cert_splits=cert_splits,
                metadata_for=metadata_for,
            )
            for block_index, rows in enumerate(target_blocks):
                block_rows[block_index].extend(rows)
        return [
            pd.DataFrame(rows).sort_values(["target", "source"]).reset_index(drop=True)
            if rows
            else pd.DataFrame()
            for rows in block_rows
        ]
    rows: list[dict] = []
    for target, group_sources in normalized_groups.items():
        rows.extend(_target_rows(driver, target, group_sources, metadata_for=metadata_for))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target", "source"]).reset_index(drop=True)


def _target_rows(driver, target: int, group_sources: tuple[int, ...], *, metadata_for=None) -> list[dict]:
    train_idx, val_idx, cert_idx = _split_indices(
        _n_design_rows(driver, driver.splits.train_fit),
        _n_design_rows(driver, driver.splits.val_alpha),
        _n_design_rows(driver, driver.splits.cert),
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

    reduced_sources = [tuple(s for s in group_sources if s != source) for source in group_sources]
    full_sources = [group_sources for _ in group_sources]
    source_only = [(source,) for source in group_sources]
    reduced, full, source_cert, y = _target_full_cert_fits_by_edge_subbatch(
        driver,
        target=int(target),
        group_sources=group_sources,
        reduced_sources=reduced_sources,
        full_sources=full_sources,
        source_only=source_only,
        train_idx=train_idx,
        val_idx=val_idx,
        cert_idx=cert_idx,
        alphas=alphas,
        dtype=gpu_dtype,
        device=gpu_device,
        variance_eps=cfg.variance_eps,
    )
    y_cert = y[cert_idx]

    delta_true = full.r2_cert - reduced.r2_cert
    survivors = [
        i
        for i, delta in enumerate(delta_true)
        if not (cfg.skip_null_on_fail and float(delta) <= float(cfg.delta_floor))
    ]
    null_by_edge: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if survivors:
        null_by_edge = _null_batches(
            driver,
            target=int(target),
            group_sources=group_sources,
            survivors=survivors,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            cert_idx=cert_idx,
            reduced_r2=reduced.r2_cert,
            alphas=alphas,
            dtype=gpu_dtype,
            device=gpu_device,
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
            y_cert=y_cert,
            source_design_cert=source_cert[edge_index],
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
            n_train_fit=int(len(train_idx)),
            n_val_alpha=int(len(val_idx)),
            n_cert=int(len(cert_idx)),
        )
        row = _result_row(result)
        metadata = metadata_for(int(target), int(source)) if metadata_for else None
        if metadata:
            row.update(metadata)
        rows.append(row)
    return rows


def _target_full_cert_fits_by_edge_subbatch(
    driver,
    *,
    target: int,
    group_sources: tuple[int, ...],
    reduced_sources,
    full_sources,
    source_only,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cert_idx: np.ndarray,
    alphas: np.ndarray,
    dtype: str,
    device: str,
    variance_eps: float,
) -> tuple[BatchedRidgeResult, BatchedRidgeResult, np.ndarray, np.ndarray]:
    reduced_parts = []
    full_parts = []
    source_cert_parts = []
    y_ref = None
    subbatch = _gpu_edge_subbatch_size(driver)
    for start, end in _edge_subbatch_ranges(len(group_sources), subbatch):
        batch_size = end - start
        values = np.repeat(driver.values[None, :, :], batch_size, axis=0)
        targets = np.full(batch_size, int(target), dtype=np.int64)
        reduced_designs, y = _design_for_all_splits(
            driver,
            values,
            targets,
            reduced_sources[start:end],
            include_own=True,
            dtype=dtype,
            device=device,
        )
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise ValueError("edge sub-batches for one target must share y")
        full_designs, _ = _design_for_all_splits(
            driver,
            values,
            targets,
            full_sources[start:end],
            include_own=True,
            dtype=dtype,
            device=device,
        )
        source_cert_parts.append(
            batched_lagged_design(
                values,
                targets,
                source_only[start:end],
                driver.lookback,
                driver.horizon,
                int(driver.splits.cert.start),
                int(driver.splits.cert.end),
                include_own=False,
                dtype=dtype,
                device=device,
            ).X
        )
        reduced_parts.append(
            batched_ridge_svd(
                reduced_designs,
                y,
                alphas,
                train_idx=train_idx,
                val_idx=val_idx,
                cert_idx=cert_idx,
                dtype=dtype,
                device=device,
                variance_eps=variance_eps,
            )
        )
        full_parts.append(
            batched_ridge_svd(
                full_designs,
                y,
                alphas,
                train_idx=train_idx,
                val_idx=val_idx,
                cert_idx=cert_idx,
                dtype=dtype,
                device=device,
                variance_eps=variance_eps,
            )
        )
        del reduced_designs, full_designs, values
    if y_ref is None:
        raise ValueError("group_sources must not be empty")
    return (
        _concat_ridge_results(reduced_parts),
        _concat_ridge_results(full_parts),
        np.concatenate(source_cert_parts, axis=0),
        y_ref,
    )


def _target_train_val_fits_by_edge_subbatch(
    driver,
    *,
    target: int,
    group_sources: tuple[int, ...],
    reduced_sources,
    full_sources,
    alphas: np.ndarray,
    dtype: str,
    device: str,
) -> tuple[BatchedRidgeFit, BatchedRidgeFit, np.ndarray, np.ndarray]:
    reduced_parts = []
    full_parts = []
    y_train_ref = None
    y_val_ref = None
    subbatch = _gpu_edge_subbatch_size(driver)
    for start, end in _edge_subbatch_ranges(len(group_sources), subbatch):
        batch_size = end - start
        values = np.repeat(driver.values[None, :, :], batch_size, axis=0)
        targets = np.full(batch_size, int(target), dtype=np.int64)
        reduced_train, y_train, reduced_val, y_val = _design_for_train_val(
            driver,
            values,
            targets,
            reduced_sources[start:end],
            include_own=True,
            dtype=dtype,
            device=device,
        )
        if y_train_ref is None:
            y_train_ref = y_train
            y_val_ref = y_val
        elif not (np.array_equal(y_train_ref, y_train) and np.array_equal(y_val_ref, y_val)):
            raise ValueError("edge sub-batches for one target must share train/val y")
        full_train, _, full_val, _ = _design_for_train_val(
            driver,
            values,
            targets,
            full_sources[start:end],
            include_own=True,
            dtype=dtype,
            device=device,
        )
        reduced_parts.append(
            batched_ridge_fit(
                X_train=reduced_train,
                y_train=y_train,
                X_val=reduced_val,
                y_val=y_val,
                alpha_grid=alphas,
                dtype=dtype,
                device=device,
            )
        )
        full_parts.append(
            batched_ridge_fit(
                X_train=full_train,
                y_train=y_train,
                X_val=full_val,
                y_val=y_val,
                alpha_grid=alphas,
                dtype=dtype,
                device=device,
            )
        )
        del reduced_train, reduced_val, full_train, full_val, values
    if y_train_ref is None or y_val_ref is None:
        raise ValueError("group_sources must not be empty")
    return _concat_ridge_fits(reduced_parts), _concat_ridge_fits(full_parts), y_train_ref, y_val_ref


def _target_cert_metrics_by_edge_subbatch(
    driver,
    *,
    target: int,
    group_sources: tuple[int, ...],
    reduced_sources,
    full_sources,
    source_only,
    cert_split,
    reduced_fit: BatchedRidgeFit,
    full_fit: BatchedRidgeFit,
    dtype: str,
    device: str,
    variance_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_cert_parts = []
    reduced_r2_parts = []
    full_r2_parts = []
    y_cert_ref = None
    subbatch = _gpu_edge_subbatch_size(driver)
    for start, end in _edge_subbatch_ranges(len(group_sources), subbatch):
        batch_size = end - start
        values = np.repeat(driver.values[None, :, :], batch_size, axis=0)
        targets = np.full(batch_size, int(target), dtype=np.int64)
        reduced_cert = _design_for_cert_split(
            driver,
            values,
            targets,
            reduced_sources[start:end],
            cert_split,
            include_own=True,
            dtype=dtype,
            device=device,
        )
        if y_cert_ref is None:
            y_cert_ref = reduced_cert.y[0]
        elif not np.array_equal(y_cert_ref, reduced_cert.y[0]):
            raise ValueError("edge sub-batches for one target must share cert y")
        full_cert = _design_for_cert_split(
            driver,
            values,
            targets,
            full_sources[start:end],
            cert_split,
            include_own=True,
            dtype=dtype,
            device=device,
        )
        source_cert_parts.append(
            batched_lagged_design(
                values,
                targets,
                source_only[start:end],
                driver.lookback,
                driver.horizon,
                int(cert_split.start),
                int(cert_split.end),
                include_own=False,
                dtype=dtype,
                device=device,
            ).X
        )
        reduced_pred = predict_batched(fit=_slice_ridge_fit(reduced_fit, start, end), X=reduced_cert.X)
        full_pred = predict_batched(fit=_slice_ridge_fit(full_fit, start, end), X=full_cert.X)
        reduced_r2_parts.append(r2_cert_score(y_cert_ref, reduced_pred, variance_eps=variance_eps))
        full_r2_parts.append(r2_cert_score(y_cert_ref, full_pred, variance_eps=variance_eps))
        del reduced_cert, full_cert, values
    if y_cert_ref is None:
        raise ValueError("group_sources must not be empty")
    reduced_r2 = np.concatenate(reduced_r2_parts, axis=0)
    full_r2 = np.concatenate(full_r2_parts, axis=0)
    return y_cert_ref, np.concatenate(source_cert_parts, axis=0), reduced_r2, full_r2 - reduced_r2


def _target_block_rows(
    driver,
    target: int,
    group_sources: tuple[int, ...],
    *,
    cert_splits,
    metadata_for=None,
) -> list[list[dict]]:
    cfg = driver.probe_config
    if cfg.alpha_rule != "val_grid":
        raise ValueError(f"GPU batched ridge supports alpha_rule='val_grid', got {cfg.alpha_rule!r}")
    if cfg.null_method != "phase":
        raise ValueError(f"GPU candidate-group backend supports null_method='phase', got {cfg.null_method!r}")
    alphas = np.asarray(cfg.alphas, dtype=np.float64)
    gpu_dtype = _gpu_dtype(driver)
    gpu_device = _gpu_device(driver)
    group_id = f"target_{int(target)}_top{len(group_sources)}"

    reduced_sources = [tuple(s for s in group_sources if s != source) for source in group_sources]
    full_sources = [group_sources for _ in group_sources]
    source_only = [(source,) for source in group_sources]
    reduced_fit, full_fit, y_train, y_val = _target_train_val_fits_by_edge_subbatch(
        driver,
        target=int(target),
        group_sources=group_sources,
        reduced_sources=reduced_sources,
        full_sources=full_sources,
        alphas=alphas,
        dtype=gpu_dtype,
        device=gpu_device,
    )

    block_state = []
    union_survivors: set[int] = set()
    for cert_split in cert_splits:
        y_cert, source_cert, reduced_r2, delta_true = _target_cert_metrics_by_edge_subbatch(
            driver,
            target=int(target),
            group_sources=group_sources,
            reduced_sources=reduced_sources,
            full_sources=full_sources,
            source_only=source_only,
            cert_split=cert_split,
            reduced_fit=reduced_fit,
            full_fit=full_fit,
            dtype=gpu_dtype,
            device=gpu_device,
            variance_eps=cfg.variance_eps,
        )
        survivors = [
            int(i)
            for i, delta in enumerate(delta_true)
            if not (cfg.skip_null_on_fail and float(delta) <= float(cfg.delta_floor))
        ]
        union_survivors.update(survivors)
        block_state.append(
            {
                "cert_split": cert_split,
                "y_cert": y_cert,
                "source_cert": source_cert,
                "reduced_r2": reduced_r2,
                "delta_true": delta_true,
                "survivors": survivors,
            }
        )

    null_fit = None
    if union_survivors:
        null_fit = _null_fit_batches(
            driver,
            target=int(target),
            group_sources=group_sources,
            survivors=sorted(union_survivors),
            y_train=y_train,
            y_val=y_val,
            alphas=alphas,
            dtype=gpu_dtype,
            device=gpu_device,
        )

    out: list[list[dict]] = []
    for state in block_state:
        null_by_edge = (
            _eval_null_block(
                driver,
                target=int(target),
                group_sources=group_sources,
                y_cert=state["y_cert"],
                cert_split=state["cert_split"],
                survivors=state["survivors"],
                reduced_r2=state["reduced_r2"],
                null_fit=null_fit,
                dtype=gpu_dtype,
                device=gpu_device,
            )
            if null_fit is not None
            else {}
        )
        out.append(
            _rows_from_block_metrics(
                driver,
                target=int(target),
                group_sources=group_sources,
                group_id=group_id,
                y_cert=state["y_cert"],
                source_cert=state["source_cert"],
                delta_true=state["delta_true"],
                null_by_edge=null_by_edge,
                reduced_alpha=reduced_fit.alpha,
                full_alpha=full_fit.alpha,
                n_train=len(y_train),
                n_val=len(y_val),
                metadata_for=metadata_for,
            )
        )
    return out


def _split_indices(n_train: int, n_val: int, n_cert: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.arange(0, int(n_train), dtype=np.int64)
    val = np.arange(int(n_train), int(n_train + n_val), dtype=np.int64)
    cert = np.arange(int(n_train + n_val), int(n_train + n_val + n_cert), dtype=np.int64)
    return train, val, cert


def _n_design_rows(driver, split) -> int:
    future = np.arange(int(split.start), int(split.end), dtype=np.int64)
    origin = future - int(driver.horizon) + 1
    return int(np.count_nonzero(origin - int(driver.lookback) >= 0))


def _design_for_all_splits(
    driver,
    values: np.ndarray,
    targets: np.ndarray,
    sources,
    *,
    include_own: bool,
    dtype: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    train = batched_lagged_design(
        values,
        targets,
        sources,
        driver.lookback,
        driver.horizon,
        int(driver.splits.train_fit.start),
        int(driver.splits.train_fit.end),
        include_own=include_own,
        dtype=dtype,
        device=device,
    )
    val = batched_lagged_design(
        values,
        targets,
        sources,
        driver.lookback,
        driver.horizon,
        int(driver.splits.val_alpha.start),
        int(driver.splits.val_alpha.end),
        include_own=include_own,
        dtype=dtype,
        device=device,
    )
    cert = batched_lagged_design(
        values,
        targets,
        sources,
        driver.lookback,
        driver.horizon,
        int(driver.splits.cert.start),
        int(driver.splits.cert.end),
        include_own=include_own,
        dtype=dtype,
        device=device,
    )
    y = np.concatenate([train.y[0], val.y[0], cert.y[0]])
    if not (
        np.allclose(train.y, train.y[0], rtol=0.0, atol=0.0)
        and np.allclose(val.y, val.y[0], rtol=0.0, atol=0.0)
        and np.allclose(cert.y, cert.y[0], rtol=0.0, atol=0.0)
    ):
        raise ValueError("batched designs for one target must share y across batch")
    return np.concatenate([train.X, val.X, cert.X], axis=1), y


def _null_batches(
    driver,
    *,
    target: int,
    group_sources: tuple[int, ...],
    survivors: list[int],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cert_idx: np.ndarray,
    reduced_r2: np.ndarray,
    alphas: np.ndarray,
    dtype: str,
    device: str,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    cfg = driver.probe_config
    pairs = [
        (int(edge_index), int(surrogate_id))
        for edge_index in survivors
        for surrogate_id in range(int(cfg.B))
    ]
    rows_total = int(len(y))
    cols_full = int((1 + len(group_sources)) * driver.lookback)
    chunk_size = _gpu_chunk_size(driver, rows_total, cols_full, dtype)
    surrogate_chunk_size = _gpu_surrogate_chunk_size(driver, chunk_size)
    delta_parts: dict[int, list[float]] = {int(edge_index): [] for edge_index in survivors}
    alpha_parts: dict[int, list[float]] = {int(edge_index): [] for edge_index in survivors}
    for start in range(0, len(pairs), min(chunk_size, surrogate_chunk_size)):
        chunk = pairs[start : start + min(chunk_size, surrogate_chunk_size)]
        values = np.repeat(driver.values[None, :, :], len(chunk), axis=0)
        source_indices = np.asarray([int(group_sources[edge_index]) for edge_index, _ in chunk], dtype=np.int64)
        surrogate_ids = np.asarray([int(surrogate_id) for _, surrogate_id in chunk], dtype=np.int64)
        surrogate_rows = batched_phase_surrogate_pairs(
            driver.values[:, source_indices].T,
            source_indices=source_indices,
            surrogate_ids=surrogate_ids,
            seed=int(driver.seed),
            dtype=dtype,
            device=device,
        )
        _assign_surrogate_rows(values, source_indices, surrogate_rows)
        targets = np.full(len(chunk), int(target), dtype=np.int64)
        null_designs, _ = _design_for_all_splits(
            driver,
            values,
            targets,
            [group_sources for _ in chunk],
            include_own=True,
            dtype=dtype,
            device=device,
        )
        null = batched_ridge_svd(
            null_designs,
            y,
            alphas,
            train_idx=train_idx,
            val_idx=val_idx,
            cert_idx=cert_idx,
            dtype=dtype,
            device=device,
            variance_eps=cfg.variance_eps,
        )
        for batch_index, (edge_index, _surrogate_id) in enumerate(chunk):
            delta_parts[int(edge_index)].append(float(null.r2_cert[batch_index] - reduced_r2[int(edge_index)]))
            alpha_parts[int(edge_index)].append(float(null.alpha[batch_index]))
    return {
        edge_index: (
            np.asarray(delta_parts[edge_index], dtype=np.float64),
            np.asarray(alpha_parts[edge_index], dtype=np.float64),
        )
        for edge_index in delta_parts
    }


def _design_for_train_val(
    driver,
    values: np.ndarray,
    targets: np.ndarray,
    sources,
    *,
    include_own: bool,
    dtype: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = batched_lagged_design(
        values,
        targets,
        sources,
        driver.lookback,
        driver.horizon,
        int(driver.splits.train_fit.start),
        int(driver.splits.train_fit.end),
        include_own=include_own,
        dtype=dtype,
        device=device,
    )
    val = batched_lagged_design(
        values,
        targets,
        sources,
        driver.lookback,
        driver.horizon,
        int(driver.splits.val_alpha.start),
        int(driver.splits.val_alpha.end),
        include_own=include_own,
        dtype=dtype,
        device=device,
    )
    if not (
        np.allclose(train.y, train.y[0], rtol=0.0, atol=0.0)
        and np.allclose(val.y, val.y[0], rtol=0.0, atol=0.0)
    ):
        raise ValueError("batched train/val designs for one target must share y across batch")
    return train.X, train.y[0], val.X, val.y[0]


def _design_for_cert_split(
    driver,
    values: np.ndarray,
    targets: np.ndarray,
    sources,
    cert_split,
    *,
    include_own: bool,
    dtype: str,
    device: str,
):
    return batched_lagged_design(
        values,
        targets,
        sources,
        driver.lookback,
        driver.horizon,
        int(cert_split.start),
        int(cert_split.end),
        include_own=include_own,
        dtype=dtype,
        device=device,
    )


def _null_fit_batches(
    driver,
    *,
    target: int,
    group_sources: tuple[int, ...],
    survivors: list[int],
    y_train: np.ndarray,
    y_val: np.ndarray,
    alphas: np.ndarray,
    dtype: str,
    device: str,
) -> dict:
    cfg = driver.probe_config
    pairs = [
        (int(edge_index), int(surrogate_id))
        for edge_index in survivors
        for surrogate_id in range(int(cfg.B))
    ]
    rows_total = int(len(y_train) + len(y_val))
    cols_full = int((1 + len(group_sources)) * driver.lookback)
    chunk_size = _gpu_chunk_size(driver, rows_total, cols_full, dtype)
    surrogate_chunk_size = _gpu_surrogate_chunk_size(driver, chunk_size)
    coefs = []
    intercepts = []
    alphas_selected = []
    ordered_pairs = []
    for start in range(0, len(pairs), min(chunk_size, surrogate_chunk_size)):
        chunk = pairs[start : start + min(chunk_size, surrogate_chunk_size)]
        values = np.repeat(driver.values[None, :, :], len(chunk), axis=0)
        source_indices = np.asarray([int(group_sources[edge_index]) for edge_index, _ in chunk], dtype=np.int64)
        surrogate_ids = np.asarray([int(surrogate_id) for _, surrogate_id in chunk], dtype=np.int64)
        surrogate_rows = batched_phase_surrogate_pairs(
            driver.values[:, source_indices].T,
            source_indices=source_indices,
            surrogate_ids=surrogate_ids,
            seed=int(driver.seed),
            dtype=dtype,
            device=device,
        )
        _assign_surrogate_rows(values, source_indices, surrogate_rows)
        targets = np.full(len(chunk), int(target), dtype=np.int64)
        train, _, val, _ = _design_for_train_val(
            driver,
            values,
            targets,
            [group_sources for _ in chunk],
            include_own=True,
            dtype=dtype,
            device=device,
        )
        fit = batched_ridge_fit(
            X_train=train,
            y_train=y_train,
            X_val=val,
            y_val=y_val,
            alpha_grid=alphas,
            dtype=dtype,
            device=device,
        )
        coefs.append(fit.coef)
        intercepts.append(fit.intercept)
        alphas_selected.append(fit.alpha)
        ordered_pairs.extend(chunk)
    return {
        "pairs": ordered_pairs,
        "coef": np.concatenate(coefs, axis=0) if coefs else np.empty((0, cols_full), dtype=np.float64),
        "intercept": np.concatenate(intercepts, axis=0) if intercepts else np.empty((0,), dtype=np.float64),
        "alpha": np.concatenate(alphas_selected, axis=0) if alphas_selected else np.empty((0,), dtype=np.float64),
    }


def _eval_null_block(
    driver,
    *,
    target: int,
    group_sources: tuple[int, ...],
    y_cert: np.ndarray,
    cert_split,
    survivors: list[int],
    reduced_r2: np.ndarray,
    null_fit: dict,
    dtype: str,
    device: str,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not survivors:
        return {}
    survivor_set = set(int(edge_index) for edge_index in survivors)
    selected = [
        (pair_index, pair)
        for pair_index, pair in enumerate(null_fit["pairs"])
        if int(pair[0]) in survivor_set
    ]
    rows_total = int(len(y_cert))
    cols_full = int((1 + len(group_sources)) * driver.lookback)
    chunk_size = _gpu_chunk_size(driver, rows_total, cols_full, dtype)
    surrogate_chunk_size = _gpu_surrogate_chunk_size(driver, chunk_size)
    delta_parts: dict[int, list[float]] = {int(edge_index): [] for edge_index in survivors}
    alpha_parts: dict[int, list[float]] = {int(edge_index): [] for edge_index in survivors}
    for start in range(0, len(selected), min(chunk_size, surrogate_chunk_size)):
        chunk = selected[start : start + min(chunk_size, surrogate_chunk_size)]
        values = np.repeat(driver.values[None, :, :], len(chunk), axis=0)
        coef = []
        intercept = []
        source_indices = np.asarray(
            [int(group_sources[edge_index]) for _pair_index, (edge_index, _surrogate_id) in chunk],
            dtype=np.int64,
        )
        surrogate_ids = np.asarray(
            [int(surrogate_id) for _pair_index, (_edge_index, surrogate_id) in chunk],
            dtype=np.int64,
        )
        surrogate_rows = batched_phase_surrogate_pairs(
            driver.values[:, source_indices].T,
            source_indices=source_indices,
            surrogate_ids=surrogate_ids,
            seed=int(driver.seed),
            dtype=dtype,
            device=device,
        )
        _assign_surrogate_rows(values, source_indices, surrogate_rows)
        for pair_index, (_edge_index, _surrogate_id) in chunk:
            coef.append(null_fit["coef"][pair_index])
            intercept.append(null_fit["intercept"][pair_index])
        targets = np.full(len(chunk), int(target), dtype=np.int64)
        cert = _design_for_cert_split(
            driver,
            values,
            targets,
            [group_sources for _ in chunk],
            cert_split,
            include_own=True,
            dtype=dtype,
            device=device,
        )
        pred = np.einsum("brc,bc->br", cert.X, np.asarray(coef)) + np.asarray(intercept)[:, None]
        r2 = r2_cert_score(y_cert, pred, variance_eps=driver.probe_config.variance_eps)
        for batch_index, (pair_index, (edge_index, _surrogate_id)) in enumerate(chunk):
            delta_parts[int(edge_index)].append(float(r2[batch_index] - reduced_r2[int(edge_index)]))
            alpha_parts[int(edge_index)].append(float(null_fit["alpha"][pair_index]))
    return {
        edge_index: (
            np.asarray(delta_parts[edge_index], dtype=np.float64),
            np.asarray(alpha_parts[edge_index], dtype=np.float64),
        )
        for edge_index in delta_parts
    }


def _rows_from_block_metrics(
    driver,
    *,
    target: int,
    group_sources: tuple[int, ...],
    group_id: str,
    y_cert: np.ndarray,
    source_cert: np.ndarray,
    delta_true: np.ndarray,
    null_by_edge: dict[int, tuple[np.ndarray, np.ndarray]],
    reduced_alpha: np.ndarray,
    full_alpha: np.ndarray,
    n_train: int,
    n_val: int,
    metadata_for=None,
) -> list[dict]:
    cfg = driver.probe_config
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
            y_cert=y_cert,
            source_design_cert=source_cert[edge_index],
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
            alpha_reduced=float(reduced_alpha[edge_index]),
            alpha_full=float(full_alpha[edge_index]),
            alpha_null=np.asarray(alpha_null, dtype=np.float64),
            alpha_rule=cfg.alpha_rule,
            gate_delta_true=gates.gate_delta_true,
            gate_aligned_gain=gates.gate_aligned_gain,
            near_zero_target_variance=gates.near_zero_target_variance,
            sparse_zero=gates.sparse_zero,
            unstable_metric=gates.unstable_metric,
            certified_candidate=gates.certified_candidate,
            skipped_null=skipped_null,
            n_train_fit=int(n_train),
            n_val_alpha=int(n_val),
            n_cert=int(len(y_cert)),
        )
        row = _result_row(result)
        metadata = metadata_for(int(target), int(source)) if metadata_for else None
        if metadata:
            row.update(metadata)
        rows.append(row)
    return rows


def _gpu_chunk_size(driver, rows: int, cols: int, dtype: str) -> int:
    raw = getattr(driver, "gpu_chunk", os.environ.get("PSVCA_GPU_CHUNK"))
    if raw is not None:
        chunk = int(raw)
        if chunk <= 0:
            raise ValueError("PSVCA_GPU_CHUNK must be positive")
        return chunk
    bytes_per = 8 if dtype == "float64" else 4
    six_gib = 6 * 1024**3
    by_matrix_bytes = max(1, int(rows) * int(cols) * bytes_per)
    return max(1, min(64, six_gib // by_matrix_bytes))


def _gpu_surrogate_chunk_size(driver, fallback: int) -> int:
    raw = getattr(driver, "gpu_surrogate_chunk", os.environ.get("PSVCA_GPU_SURROGATE_CHUNK"))
    if raw is None:
        return max(1, int(fallback))
    chunk = int(raw)
    if chunk <= 0:
        raise ValueError("PSVCA_GPU_SURROGATE_CHUNK must be positive")
    return chunk


def _gpu_edge_subbatch_size(driver) -> int:
    raw = getattr(driver, "gpu_edge_subbatch", os.environ.get("PSVCA_GPU_EDGE_SUBBATCH", 4))
    subbatch = int(raw)
    if subbatch <= 0:
        raise ValueError("PSVCA_GPU_EDGE_SUBBATCH must be positive")
    return subbatch


def _edge_subbatch_ranges(n_edges: int, subbatch: int):
    for start in range(0, int(n_edges), int(subbatch)):
        yield start, min(start + int(subbatch), int(n_edges))


def _assign_surrogate_rows(values: np.ndarray, source_cols: np.ndarray, surrogate_rows: np.ndarray) -> None:
    cols = np.asarray(source_cols, dtype=np.int64)
    rows = np.asarray(surrogate_rows)
    if values.shape[0] != cols.shape[0] or rows.shape[0] != cols.shape[0]:
        raise ValueError("surrogate fill batch dimensions differ")
    values[np.arange(cols.shape[0]), :, cols] = rows


def _concat_ridge_results(parts: list[BatchedRidgeResult]) -> BatchedRidgeResult:
    if not parts:
        raise ValueError("cannot concatenate empty ridge result list")
    return BatchedRidgeResult(
        coef=np.concatenate([part.coef for part in parts], axis=0),
        intercept=np.concatenate([part.intercept for part in parts], axis=0),
        alpha_idx=np.concatenate([part.alpha_idx for part in parts], axis=0),
        alpha=np.concatenate([part.alpha for part in parts], axis=0),
        r2_cert=np.concatenate([part.r2_cert for part in parts], axis=0),
        pred_cert=np.concatenate([part.pred_cert for part in parts], axis=0),
    )


def _concat_ridge_fits(parts: list[BatchedRidgeFit]) -> BatchedRidgeFit:
    if not parts:
        raise ValueError("cannot concatenate empty ridge fit list")
    return BatchedRidgeFit(
        coef=np.concatenate([part.coef for part in parts], axis=0),
        intercept=np.concatenate([part.intercept for part in parts], axis=0),
        alpha_idx=np.concatenate([part.alpha_idx for part in parts], axis=0),
        alpha=np.concatenate([part.alpha for part in parts], axis=0),
    )


def _slice_ridge_fit(fit: BatchedRidgeFit, start: int, end: int) -> BatchedRidgeFit:
    return BatchedRidgeFit(
        coef=fit.coef[start:end],
        intercept=fit.intercept[start:end],
        alpha_idx=fit.alpha_idx[start:end],
        alpha=fit.alpha[start:end],
    )


def _gpu_dtype(driver) -> str:
    raw = str(getattr(driver, "gpu_dtype", os.environ.get("PSVCA_GPU_DTYPE", "float32"))).lower()
    if raw in {"fp64", "float64", "double"}:
        return "float64"
    if raw in {"fp32", "float32", "single"}:
        return "float32"
    raise ValueError(f"unsupported GPU dtype: {raw!r}")


def _gpu_device(driver) -> str:
    return resolve_gpu_device(driver=driver)


def _result_row(result: CandidateGroupProbeResult) -> dict:
    data = asdict(result)
    data.pop("delta_null", None)
    data.pop("alpha_null", None)
    return data
