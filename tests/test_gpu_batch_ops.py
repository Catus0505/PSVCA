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
from psvca.gpu.batched_surrogate import batched_phase_surrogate
from psvca.gpu.driver_gpu import run_gpu_batch
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
