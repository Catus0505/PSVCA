from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.gpu.batched_design import batched_lagged_design
from psvca.gpu.device import resolve_gpu_device
from psvca.gpu.batched_surrogate import batched_phase_surrogate, batched_phase_surrogate_pairs
from psvca.gpu.driver_gpu import (
    _assign_surrogate_rows,
    _n_design_rows,
    _split_indices,
    _target_full_cert_fits_by_edge_subbatch,
    run_gpu_batch,
)
from psvca.linalg.design import make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate
from psvca.certify.probe import PairwiseProbeConfig
from psvca.pipeline.driver import CertificationDriver
from psvca.pipeline.reference import _cert_blocks
from scripts.run_synthetic_check import full_splits, make_planted_values


pytest.importorskip("torch")


def test_resolve_gpu_device_priority(monkeypatch) -> None:
    monkeypatch.delenv("PSVCA_GPU_DEVICE", raising=False)
    assert resolve_gpu_device() == "cuda:0"

    monkeypatch.setenv("PSVCA_GPU_DEVICE", "cuda:1")
    assert resolve_gpu_device() == "cuda:1"
    assert resolve_gpu_device(driver=SimpleNamespace(gpu_device="cpu")) == "cpu"
    assert resolve_gpu_device(driver=SimpleNamespace(gpu_device=None)) == "cuda:1"
    assert resolve_gpu_device(driver=SimpleNamespace(gpu_device="cuda:2"), device="cpu") == "cpu"


def test_batched_lagged_design_matches_cpu_fp64() -> None:
    rng = np.random.default_rng(123)
    values = rng.normal(size=(3, 20, 5))
    targets = np.asarray([0, 1, 2])
    sources = [(1, 2), (0, 3), (3, 4)]

    actual = batched_lagged_design(
        values,
        targets,
        sources,
        lookback=4,
        horizon=2,
        y_start=1,
        y_end=18,
        include_own=True,
        dtype="float64",
        device="cpu",
        assert_cpu_equiv=True,
    )

    for i in range(values.shape[0]):
        expected = make_lagged_design(
            values[i],
            int(targets[i]),
            sources[i],
            lookback=4,
            horizon=2,
            y_start=1,
            y_end=18,
            include_own=True,
        )
        np.testing.assert_array_equal(actual.X[i], expected.X)
        np.testing.assert_array_equal(actual.y[i], expected.y)
        np.testing.assert_array_equal(actual.future_indices, expected.future_indices)
        np.testing.assert_array_equal(actual.origin_indices, expected.origin_indices)


def test_batched_phase_surrogate_matches_cpu_and_source_seed_order() -> None:
    rng = np.random.default_rng(456)
    values = rng.normal(size=(3, 31))
    source_indices = np.asarray([4, 1, 7])
    B = 4
    seed = 2026

    actual = batched_phase_surrogate(
        values,
        source_indices=source_indices,
        B=B,
        seed=seed,
        dtype="float64",
        device="cpu",
        assert_cpu_equiv=True,
    )
    for source_pos, source_idx in enumerate(source_indices):
        for surrogate_id in range(B):
            expected = make_phase_surrogate(
                values[source_pos],
                source_idx=int(source_idx),
                surrogate_id=surrogate_id,
                seed=seed,
                dataset="gpu_batch_ops",
                split="pre_test",
                cache_dir=None,
            ).values
            np.testing.assert_allclose(actual[source_pos, surrogate_id], expected, rtol=1e-12, atol=1e-12)

    perm = np.asarray([2, 0, 1])
    reordered = batched_phase_surrogate(
        values[perm],
        source_indices=source_indices[perm],
        B=B,
        seed=seed,
        dtype="float64",
        device="cpu",
    )
    np.testing.assert_allclose(reordered[np.argsort(perm)], actual, rtol=1e-12, atol=1e-12)


def test_batched_phase_surrogate_pairs_matches_full_grid_and_cpu() -> None:
    rng = np.random.default_rng(789)
    source_series = rng.normal(size=(2, 29))
    source_indices = np.asarray([3, 8])
    B = 5
    seed = 2026
    full = batched_phase_surrogate(
        source_series,
        source_indices=source_indices,
        B=B,
        seed=seed,
        dtype="float64",
        device="cpu",
    )
    pairs = [(0, 0), (0, 3), (1, 1), (0, 4), (1, 4), (1, 2), (0, 2)]
    pair_values = source_series[[source_pos for source_pos, _sid in pairs]]
    pair_sources = source_indices[[source_pos for source_pos, _sid in pairs]]
    pair_ids = np.asarray([sid for _source_pos, sid in pairs])
    actual = batched_phase_surrogate_pairs(
        pair_values,
        source_indices=pair_sources,
        surrogate_ids=pair_ids,
        seed=seed,
        dtype="float64",
        device="cpu",
        assert_cpu_equiv=True,
    )
    for pair_pos, (source_pos, surrogate_id) in enumerate(pairs):
        np.testing.assert_allclose(
            actual[pair_pos],
            full[source_pos, surrogate_id],
            rtol=1e-12,
            atol=1e-12,
        )


def test_vectorized_surrogate_fill_matches_python_loop() -> None:
    rng = np.random.default_rng(321)
    values_loop = np.zeros((7, 11, 5), dtype=np.float64)
    values_vec = values_loop.copy()
    source_cols = np.asarray([3, 1, 4, 0, 2, 3, 1], dtype=np.int64)
    surrogate_rows = rng.normal(size=(7, 11))

    for batch_index, source_col in enumerate(source_cols):
        values_loop[batch_index, :, int(source_col)] = surrogate_rows[batch_index]
    _assign_surrogate_rows(values_vec, source_cols, surrogate_rows)

    np.testing.assert_array_equal(values_vec, values_loop)


def test_edge_subbatch_design_and_svd_match_full_batch() -> None:
    rng = np.random.default_rng(654)
    values = rng.normal(size=(360, 21))
    group_sources = tuple(range(1, 21))
    cfg = PairwiseProbeConfig(
        alphas=(0.01, 0.1, 1.0),
        B=2,
        seed=2026,
        null_method="phase",
        alpha_rule="val_grid",
        skip_null_on_fail=False,
    )
    driver = CertificationDriver(
        values=values,
        splits=full_splits(),
        lookback=6,
        horizon=1,
        probe_config=cfg,
        seed=2026,
        dataset="gpu_batch_ops",
        backend="gpu",
        gpu_device="cpu",
    )
    driver.gpu_dtype = "float64"
    train_idx, val_idx, cert_idx = _split_indices(
        _n_design_rows(driver, driver.splits.train_fit),
        _n_design_rows(driver, driver.splits.val_alpha),
        _n_design_rows(driver, driver.splits.cert),
    )
    reduced_sources = [tuple(s for s in group_sources if s != source) for source in group_sources]
    full_sources = [group_sources for _ in group_sources]
    source_only = [(source,) for source in group_sources]

    driver.gpu_edge_subbatch = 64
    reference = _target_full_cert_fits_by_edge_subbatch(
        driver,
        target=0,
        group_sources=group_sources,
        reduced_sources=reduced_sources,
        full_sources=full_sources,
        source_only=source_only,
        train_idx=train_idx,
        val_idx=val_idx,
        cert_idx=cert_idx,
        alphas=np.asarray(cfg.alphas, dtype=np.float64),
        dtype="float64",
        device="cpu",
        variance_eps=cfg.variance_eps,
    )

    for subbatch in (4, 3):
        driver.gpu_edge_subbatch = subbatch
        actual = _target_full_cert_fits_by_edge_subbatch(
            driver,
            target=0,
            group_sources=group_sources,
            reduced_sources=reduced_sources,
            full_sources=full_sources,
            source_only=source_only,
            train_idx=train_idx,
            val_idx=val_idx,
            cert_idx=cert_idx,
            alphas=np.asarray(cfg.alphas, dtype=np.float64),
            dtype="float64",
            device="cpu",
            variance_eps=cfg.variance_eps,
        )
        for actual_ridge, expected_ridge in ((actual[0], reference[0]), (actual[1], reference[1])):
            np.testing.assert_allclose(actual_ridge.coef, expected_ridge.coef, atol=1e-12, rtol=1e-9)
            np.testing.assert_allclose(
                actual_ridge.intercept,
                expected_ridge.intercept,
                atol=1e-12,
                rtol=1e-9,
            )
            np.testing.assert_array_equal(actual_ridge.alpha_idx, expected_ridge.alpha_idx)
            np.testing.assert_allclose(actual_ridge.alpha, expected_ridge.alpha, atol=1e-12, rtol=1e-9)
            np.testing.assert_allclose(
                actual_ridge.r2_cert,
                expected_ridge.r2_cert,
                atol=1e-12,
                rtol=1e-9,
            )
            np.testing.assert_allclose(
                actual_ridge.pred_cert,
                expected_ridge.pred_cert,
                atol=1e-12,
                rtol=1e-9,
            )
        np.testing.assert_allclose(actual[2], reference[2], atol=1e-12, rtol=1e-9)
        np.testing.assert_array_equal(actual[3], reference[3])


def test_gpu_edge_subbatch_preserves_candidate_group_results() -> None:
    values = make_planted_values(seed=2026)
    cfg = PairwiseProbeConfig(
        alphas=(0.01, 0.1, 1.0),
        B=4,
        seed=2026,
        null_method="phase",
        alpha_rule="val_grid",
        skip_null_on_fail=False,
    )
    target_groups = {0: (1, 2, 3)}

    def run_with_edge_subbatch(subbatch: int):
        driver = CertificationDriver(
            values=values,
            splits=full_splits(),
            lookback=6,
            horizon=1,
            probe_config=cfg,
            seed=2026,
            dataset="gpu_batch_ops",
            backend="gpu",
            gpu_device="cpu",
        )
        driver.gpu_dtype = "float64"
        driver.gpu_chunk = 3
        driver.gpu_edge_subbatch = subbatch
        return driver.candidate_group_edges(target_groups)

    reference = run_with_edge_subbatch(64)
    chunked = run_with_edge_subbatch(2)
    assert chunked[["target", "source"]].equals(reference[["target", "source"]])
    np.testing.assert_array_equal(
        chunked["certified_candidate"].to_numpy(bool),
        reference["certified_candidate"].to_numpy(bool),
    )
    np.testing.assert_array_equal(
        chunked["p_value"].to_numpy(float),
        reference["p_value"].to_numpy(float),
    )


def test_gpu_cert_blocks_reuse_fit_matches_per_block_gpu() -> None:
    values = make_planted_values(seed=2026)
    splits = full_splits()
    blocks = _cert_blocks(splits, 3)
    target_groups = {0: (1, 2, 3)}
    cfg = PairwiseProbeConfig(
        alphas=(0.01, 0.1, 1.0),
        B=3,
        seed=2026,
        null_method="phase",
        alpha_rule="val_grid",
        skip_null_on_fail=False,
    )
    driver = CertificationDriver(
        values=values,
        splits=splits,
        lookback=6,
        horizon=1,
        probe_config=cfg,
        seed=2026,
        dataset="gpu_batch_ops",
        backend="gpu",
        gpu_device="cpu",
    )
    driver.gpu_dtype = "float64"
    driver.gpu_chunk = 2
    batched = run_gpu_batch(
        driver=driver,
        target_groups=target_groups,
        cert_splits=tuple(block.cert for block in blocks),
    )
    for block_index, block in enumerate(blocks):
        per_block = CertificationDriver(
            values=values,
            splits=block,
            lookback=6,
            horizon=1,
            probe_config=cfg,
            seed=2026,
            dataset="gpu_batch_ops",
            backend="gpu",
            gpu_device="cpu",
        )
        per_block.gpu_dtype = "float64"
        per_block.gpu_chunk = 2
        expected = per_block.candidate_group_edges(target_groups)
        actual = batched[block_index]
        assert actual[["target", "source"]].equals(expected[["target", "source"]])
        np.testing.assert_array_equal(
            actual["certified_candidate"].to_numpy(bool),
            expected["certified_candidate"].to_numpy(bool),
        )
        np.testing.assert_array_equal(
            actual["p_value"].to_numpy(float),
            expected["p_value"].to_numpy(float),
        )


def test_gpu_surrogate_chunking_preserves_candidate_group_results() -> None:
    values = make_planted_values(seed=2026)
    cfg = PairwiseProbeConfig(
        alphas=(0.01, 0.1, 1.0),
        B=5,
        seed=2026,
        null_method="phase",
        alpha_rule="val_grid",
        skip_null_on_fail=False,
    )
    target_groups = {0: (1, 2, 3)}

    def run_with_surrogate_chunk(chunk: int):
        driver = CertificationDriver(
            values=values,
            splits=full_splits(),
            lookback=6,
            horizon=1,
            probe_config=cfg,
            seed=2026,
            dataset="gpu_batch_ops",
            backend="gpu",
            gpu_device="cpu",
        )
        driver.gpu_dtype = "float64"
        driver.gpu_chunk = 4
        driver.gpu_surrogate_chunk = chunk
        return driver.candidate_group_edges(target_groups)

    reference = run_with_surrogate_chunk(64)
    chunked = run_with_surrogate_chunk(3)
    assert chunked[["target", "source"]].equals(reference[["target", "source"]])
    np.testing.assert_array_equal(
        chunked["certified_candidate"].to_numpy(bool),
        reference["certified_candidate"].to_numpy(bool),
    )
    np.testing.assert_array_equal(
        chunked["p_value"].to_numpy(float),
        reference["p_value"].to_numpy(float),
    )
