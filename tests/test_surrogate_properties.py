from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.nulls.phase_surrogate import (
    make_phase_surrogate,
    make_surrogate_bank_for_source,
    phase_randomize_1d,
)
from psvca.nulls.row_perm import row_permute_1d


def _ar1_signal(n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    noise = rng.normal(scale=0.25, size=n)
    source = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        source[t] = 0.75 * source[t - 1] + noise[t]
    source += 0.8 * np.sin(np.arange(n) * 2.0 * np.pi / 29.0)
    other = np.roll(source, 3) + 0.05 * rng.normal(size=n)
    return source, other


def _circular_autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    centered = x - x.mean()
    spectrum = np.fft.rfft(centered)
    acov = np.fft.irfft(np.abs(spectrum) ** 2, n=len(centered))
    return acov[: max_lag + 1] / acov[0]


def _max_abs_cross_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    x0 = (x - x.mean()) / x.std()
    y0 = (y - y.mean()) / y.std()
    vals = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            vals.append(abs(float(np.mean(x0[:lag] * y0[-lag:]))))
        elif lag > 0:
            vals.append(abs(float(np.mean(x0[lag:] * y0[:-lag]))))
        else:
            vals.append(abs(float(np.mean(x0 * y0))))
    return max(vals)


def test_phase_surrogate_preserves_power_spectrum_and_autocorr() -> None:
    source, _ = _ar1_signal()
    surrogate = phase_randomize_1d(source, seed=5)
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(surrogate)),
        np.abs(np.fft.rfft(source)),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(surrogate.mean(), source.mean(), atol=1e-12)
    np.testing.assert_allclose(
        _circular_autocorr(surrogate, 40),
        _circular_autocorr(source, 40),
        rtol=1e-10,
        atol=1e-10,
    )


def test_phase_surrogate_reduces_cross_correlation_with_other_series() -> None:
    source, other = _ar1_signal()
    surrogate = make_phase_surrogate(
        source,
        source_idx=2,
        surrogate_id=0,
        seed=99,
        dataset="unit",
        split="pre_test",
    ).values
    original_cc = _max_abs_cross_corr(source, other, max_lag=8)
    surrogate_cc = _max_abs_cross_corr(surrogate, other, max_lag=8)
    assert original_cc > 0.85
    assert surrogate_cc < original_cc * 0.70


def test_phase_surrogate_is_reproducible_and_surrogate_id_changes_result() -> None:
    source, _ = _ar1_signal()
    a = make_phase_surrogate(source, source_idx=1, surrogate_id=3, seed=7).values
    b = make_phase_surrogate(source, source_idx=1, surrogate_id=3, seed=7).values
    c = make_phase_surrogate(source, source_idx=1, surrogate_id=4, seed=7).values
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_phase_surrogate_cache_second_call_hits_and_key_has_no_target(tmp_path: Path) -> None:
    source, _ = _ar1_signal()
    first = make_phase_surrogate(
        source,
        source_idx=0,
        surrogate_id=2,
        seed=11,
        dataset="smoke",
        split="pre_test",
        cache_dir=tmp_path,
    )
    second = make_phase_surrogate(
        source,
        source_idx=0,
        surrogate_id=2,
        seed=11,
        dataset="smoke",
        split="pre_test",
        cache_dir=tmp_path,
    )
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.cache_path == first.cache_path
    assert first.cache_path is not None
    assert "target" not in first.cache_path.name.lower()
    np.testing.assert_array_equal(first.values, second.values)


def test_surrogate_bank_for_source_uses_source_keys_only(tmp_path: Path) -> None:
    source, _ = _ar1_signal()
    bank = make_surrogate_bank_for_source(
        source,
        source_idx=4,
        B=3,
        seed=13,
        dataset="unit",
        split="pre_test",
        cache_dir=tmp_path,
    )
    assert len(bank) == 3
    assert [result.key.surrogate_id for result in bank] == [0, 1, 2]
    assert all(result.key.source_idx == 4 for result in bank)


def test_constant_series_returns_copy() -> None:
    x = np.ones(16)
    surrogate = phase_randomize_1d(x, seed=0)
    assert surrogate is not x
    np.testing.assert_array_equal(surrogate, x)


def test_row_permutation_preserves_multiset_and_is_reproducible() -> None:
    x = np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0])
    a = row_permute_1d(x, seed=42)
    b = row_permute_1d(x, seed=42)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.sort(a), np.sort(x))
