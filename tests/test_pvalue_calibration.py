from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kstest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.certify.gates import evaluate_pairwise_gates
from psvca.certify.probe import PairwiseProbeConfig, probe_pairwise
from psvca.linalg.design import make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate


@dataclass(frozen=True)
class Split:
    start: int
    end: int


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    x = np.zeros(n, dtype=np.float64)
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def _make_null_values(seed: int, n: int = 260) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = _ar1(rng, n, phi=0.72)
    source = _ar1(rng, n, phi=0.68)
    # target uses only own dynamics; source remains independent but autocorrelated.
    target = target + 0.3 * np.sin(np.arange(n) * 2.0 * np.pi / 37.0)
    source = source + 0.4 * np.cos(np.arange(n) * 2.0 * np.pi / 29.0)
    return np.column_stack([target, source])


def _own_block(values: np.ndarray, split: Split, *, lookback: int, horizon: int):
    return make_lagged_design(
        values,
        target=0,
        sources=(),
        lookback=lookback,
        horizon=horizon,
        y_start=split.start,
        y_end=split.end,
        include_own=True,
    )


def _source_block(values: np.ndarray, split: Split, *, lookback: int, horizon: int):
    return make_lagged_design(
        values,
        target=0,
        sources=(1,),
        lookback=lookback,
        horizon=horizon,
        y_start=split.start,
        y_end=split.end,
        include_own=False,
    )


def _probe_for_seed(seed: int, *, B: int):
    values = _make_null_values(seed)
    lookback = 5
    horizon = 1
    train = Split(40, 120)
    val = Split(120, 180)
    cert = Split(180, 250)
    own_train = _own_block(values, train, lookback=lookback, horizon=horizon)
    own_val = _own_block(values, val, lookback=lookback, horizon=horizon)
    own_cert = _own_block(values, cert, lookback=lookback, horizon=horizon)
    source_train = _source_block(values, train, lookback=lookback, horizon=horizon)
    source_val = _source_block(values, val, lookback=lookback, horizon=horizon)
    source_cert = _source_block(values, cert, lookback=lookback, horizon=horizon)

    bank = []
    for b in range(B):
        surrogate = make_phase_surrogate(
            values[:, 1],
            source_idx=1,
            surrogate_id=b,
            seed=seed,
            dataset="unit",
            split="pre_test",
        ).values
        s_values = values.copy()
        s_values[:, 1] = surrogate
        bank.append(
            (
                _source_block(s_values, train, lookback=lookback, horizon=horizon).X,
                _source_block(s_values, val, lookback=lookback, horizon=horizon).X,
                _source_block(s_values, cert, lookback=lookback, horizon=horizon).X,
            )
        )

    return probe_pairwise(
        target=0,
        source=1,
        y_train=own_train.y,
        y_val=own_val.y,
        y_cert=own_cert.y,
        own_train=own_train.X,
        own_val=own_val.X,
        own_cert=own_cert.X,
        source_train=source_train.X,
        source_val=source_val.X,
        source_cert=source_cert.X,
        surrogate_bank=bank,
        config=PairwiseProbeConfig(
            alphas=(0.1, 1.0, 10.0, 100.0),
            B=B,
            seed=seed,
            null_method="phase",
            alpha_rule="val_grid",
        ),
    )


def test_null_pvalues_are_calibrated_and_not_inflated() -> None:
    B = 79
    results = [_probe_for_seed(1000 + i, B=B) for i in range(45)]
    p_values = np.array([r.p_value for r in results])

    assert np.all(np.isfinite(p_values))
    assert np.all(p_values >= 1.0 / (B + 1))
    assert np.all(p_values <= 1.0)
    assert kstest(p_values, "uniform").pvalue > 0.01
    assert 0.35 <= float(np.mean(p_values)) <= 0.65
    assert float(np.mean(p_values <= 0.05)) <= 0.12
    assert float(np.mean([r.certified_candidate and r.p_value <= 0.05 for r in results])) <= 0.12
    assert all(len(r.alpha_null) == B for r in results)
    assert all(r.B == B for r in results)


def test_phase_null_requires_source_level_surrogate_bank() -> None:
    values = _make_null_values(2027, n=120)
    lookback = 5
    horizon = 1
    train = Split(20, 55)
    val = Split(55, 85)
    cert = Split(85, 115)
    own_train = _own_block(values, train, lookback=lookback, horizon=horizon)
    own_val = _own_block(values, val, lookback=lookback, horizon=horizon)
    own_cert = _own_block(values, cert, lookback=lookback, horizon=horizon)
    source_train = _source_block(values, train, lookback=lookback, horizon=horizon)
    source_val = _source_block(values, val, lookback=lookback, horizon=horizon)
    source_cert = _source_block(values, cert, lookback=lookback, horizon=horizon)

    import pytest

    with pytest.raises(ValueError, match="source-level surrogate_bank"):
        probe_pairwise(
            target=0,
            source=1,
            y_train=own_train.y,
            y_val=own_val.y,
            y_cert=own_cert.y,
            own_train=own_train.X,
            own_val=own_val.X,
            own_cert=own_cert.X,
            source_train=source_train.X,
            source_val=source_val.X,
            source_cert=source_cert.X,
            surrogate_bank=None,
            config=PairwiseProbeConfig(
                alphas=(0.1, 1.0, 10.0),
                B=5,
                seed=2027,
                null_method="phase",
                alpha_rule="val_grid",
            ),
        )


def test_nonfinite_delta_true_makes_pvalue_nan() -> None:
    n = 20
    y = np.ones(n)
    own = np.column_stack([np.linspace(0.0, 1.0, n), np.ones(n)])
    source = np.column_stack([np.sin(np.arange(n)), np.cos(np.arange(n))])
    bank = [(source.copy(), source.copy(), source.copy()) for _ in range(5)]

    result = probe_pairwise(
        target=0,
        source=1,
        y_train=y.copy(),
        y_val=y.copy(),
        y_cert=y.copy(),
        own_train=own,
        own_val=own,
        own_cert=own,
        source_train=source,
        source_val=source,
        source_cert=source,
        surrogate_bank=bank,
        config=PairwiseProbeConfig(
            alphas=(0.1, 1.0, 10.0),
            B=5,
            seed=99,
            null_method="phase",
            alpha_rule="val_grid",
        ),
    )

    assert not np.isfinite(result.delta_true)
    assert np.isnan(result.p_value)
    assert result.near_zero_target_variance
    assert result.unstable_metric
    assert not result.certified_candidate


def test_near_zero_target_variance_guard_blocks_candidate() -> None:
    y = np.ones(20)
    source = np.ones((20, 2))
    gate = evaluate_pairwise_gates(
        delta_true=1.0,
        aligned_gain=1.0,
        y_cert=y,
        source_design_cert=source,
    )
    assert gate.near_zero_target_variance
    assert gate.unstable_metric
    assert not gate.certified_candidate


def test_sparse_source_guard_blocks_candidate() -> None:
    y = np.linspace(0.0, 1.0, 20)
    source = np.zeros((20, 2))
    gate = evaluate_pairwise_gates(
        delta_true=1.0,
        aligned_gain=1.0,
        y_cert=y,
        source_design_cert=source,
    )
    assert gate.sparse_zero
    assert gate.unstable_metric
    assert not gate.certified_candidate


def test_nonfinite_metrics_trigger_unstable_guard() -> None:
    y = np.linspace(0.0, 1.0, 20)
    source = np.ones((20, 2))
    gate = evaluate_pairwise_gates(
        delta_true=float("nan"),
        aligned_gain=1.0,
        y_cert=y,
        source_design_cert=source,
    )
    assert gate.unstable_metric
    assert not gate.certified_candidate
