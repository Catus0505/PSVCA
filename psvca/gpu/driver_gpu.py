from __future__ import annotations

from dataclasses import asdict
import os

import numpy as np
import pandas as pd

from psvca.certify.gates import GateConfig, evaluate_pairwise_gates
from psvca.certify.probe import CandidateGroupProbeResult
from psvca.gpu.batched_design import batched_lagged_design
from psvca.gpu.device import resolve_gpu_device
from psvca.gpu.batched_ridge import batched_ridge_fit, batched_ridge_svd, predict_batched, r2_cert_score
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
    n_edges = len(group_sources)
    base_values = np.repeat(driver.values[None, :, :], n_edges, axis=0)
    targets = np.full(n_edges, int(target), dtype=np.int64)
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
    reduced_designs, y = _design_for_all_splits(
        driver,
        base_values,
        targets,
        reduced_sources,
        include_own=True,
        dtype=gpu_dtype,
        device=gpu_device,
    )
    y_cert = y[cert_idx]
    full_designs, _ = _design_for_all_splits(
        driver,
        base_values,
        targets,
        full_sources,
        include_own=True,
        dtype=gpu_dtype,
        device=gpu_device,
    )
    source_cert = batched_lagged_design(
        base_values,
        targets,
        source_only,
        driver.lookback,
        driver.horizon,
        int(driver.splits.cert.start),
        int(driver.splits.cert.end),
        include_own=False,
        dtype=gpu_dtype,
        device=gpu_device,
    ).X

    reduced = batched_ridge_svd(
        reduced_designs,
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
        full_designs,
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


def _target_block_rows(
    driver,
    target: int,
    group_sources: tuple[int, ...],
    *,
    cert_splits,
    metadata_for=None,
) -> list[list[dict]]:
    n_edges = len(group_sources)
    base_values = np.repeat(driver.values[None, :, :], n_edges, axis=0)
    targets = np.full(n_edges, int(target), dtype=np.int64)
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
    reduced_train, y_train, reduced_val, y_val = _design_for_train_val(
        driver,
        base_values,
        targets,
        reduced_sources,
        include_own=True,
        dtype=gpu_dtype,
        device=gpu_device,
    )
    full_train, _, full_val, _ = _design_for_train_val(
        driver,
        base_values,
        targets,
        full_sources,
        include_own=True,
        dtype=gpu_dtype,
        device=gpu_device,
    )
    reduced_fit = batched_ridge_fit(
        X_train=reduced_train,
        y_train=y_train,
        X_val=reduced_val,
        y_val=y_val,
        alpha_grid=alphas,
        dtype=gpu_dtype,
        device=gpu_device,
    )
    full_fit = batched_ridge_fit(
        X_train=full_train,
        y_train=y_train,
        X_val=full_val,
        y_val=y_val,
        alpha_grid=alphas,
        dtype=gpu_dtype,
        device=gpu_device,
    )

    block_state = []
    union_survivors: set[int] = set()
    for cert_split in cert_splits:
        reduced_cert = _design_for_cert_split(
            driver,
            base_values,
            targets,
            reduced_sources,
            cert_split,
            include_own=True,
            dtype=gpu_dtype,
            device=gpu_device,
        )
        full_cert = _design_for_cert_split(
            driver,
            base_values,
            targets,
            full_sources,
            cert_split,
            include_own=True,
            dtype=gpu_dtype,
            device=gpu_device,
        )
        source_cert = batched_lagged_design(
            base_values,
            targets,
            source_only,
            driver.lookback,
            driver.horizon,
            int(cert_split.start),
            int(cert_split.end),
            include_own=False,
            dtype=gpu_dtype,
            device=gpu_device,
        )
        y_cert = reduced_cert.y[0]
        reduced_pred = predict_batched(fit=reduced_fit, X=reduced_cert.X)
        full_pred = predict_batched(fit=full_fit, X=full_cert.X)
        reduced_r2 = r2_cert_score(y_cert, reduced_pred, variance_eps=cfg.variance_eps)
        full_r2 = r2_cert_score(y_cert, full_pred, variance_eps=cfg.variance_eps)
        delta_true = full_r2 - reduced_r2
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
                "source_cert": source_cert.X,
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
        for batch_index, (edge_index, _surrogate_id) in enumerate(chunk):
            values[batch_index, :, int(group_sources[edge_index])] = surrogate_rows[batch_index]
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
        for batch_index, (edge_index, _surrogate_id) in enumerate(chunk):
            values[batch_index, :, int(group_sources[edge_index])] = surrogate_rows[batch_index]
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
        for batch_index, (pair_index, (edge_index, _surrogate_id)) in enumerate(chunk):
            values[batch_index, :, int(group_sources[edge_index])] = surrogate_rows[batch_index]
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
