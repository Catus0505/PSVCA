from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.gpu.batched_design import batched_lagged_design
from psvca.gpu.batched_surrogate import batched_phase_surrogate
from psvca.linalg.design import make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate


pytest.importorskip("torch")


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
