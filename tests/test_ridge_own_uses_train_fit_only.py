from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.base import ridge_own as ridge_own_module
from psvca.base.innovation import compute_innovation_for_target


@dataclass(frozen=True)
class RangeLike:
    start: int
    end: int


@dataclass(frozen=True)
class SplitsLike:
    train_fit: RangeLike
    val_alpha: RangeLike
    cert: RangeLike
    original_test: RangeLike


def _series(cert_shift: float = 0.0) -> np.ndarray:
    t = np.arange(80, dtype=np.float64)
    x0 = np.sin(t / 5.0) + 0.1 * t
    x1 = np.cos(t / 7.0)
    values = np.column_stack([x0, x1])
    values[55:70, 0] += cert_shift
    return values


def test_ridge_own_fit_receives_train_fit_design_only(monkeypatch) -> None:
    splits = SplitsLike(
        train_fit=RangeLike(8, 35),
        val_alpha=RangeLike(35, 55),
        cert=RangeLike(55, 70),
        original_test=RangeLike(70, 80),
    )
    calls = []
    original_fit_ridge_path = ridge_own_module.fit_ridge_path

    def spy_fit_ridge_path(X, y, alphas):
        calls.append((np.array(X, copy=True), np.array(y, copy=True), np.array(alphas, copy=True)))
        return original_fit_ridge_path(X, y, alphas)

    monkeypatch.setattr(ridge_own_module, "fit_ridge_path", spy_fit_ridge_path)
    result = compute_innovation_for_target(
        values=_series(),
        channels=("x0", "x1"),
        splits=splits,
        target=0,
        lookback=4,
        horizon=2,
        alphas=np.array([0.01, 0.1, 1.0]),
        seed=0,
    )

    assert len(calls) == 1
    X_fit, y_fit, _ = calls[0]
    assert X_fit.shape[0] == len(result.train_fit_indices)
    assert y_fit.shape[0] == len(result.train_fit_indices)
    assert np.array_equal(y_fit, result.y_train_fit)
    assert len(result.val_alpha_indices) == len(result.pred_val_alpha)
    assert len(result.cert_indices) == len(result.pred_cert)


def test_changing_cert_values_does_not_change_fitted_own_model() -> None:
    splits = SplitsLike(
        train_fit=RangeLike(8, 35),
        val_alpha=RangeLike(35, 55),
        cert=RangeLike(55, 70),
        original_test=RangeLike(70, 80),
    )
    kwargs = dict(
        channels=("x0", "x1"),
        splits=splits,
        target=0,
        lookback=4,
        horizon=2,
        alphas=np.array([0.01, 0.1, 1.0]),
        seed=0,
    )
    result_a = compute_innovation_for_target(values=_series(cert_shift=0.0), **kwargs)
    result_b = compute_innovation_for_target(values=_series(cert_shift=100.0), **kwargs)

    assert result_a.alpha_own == result_b.alpha_own
    np.testing.assert_allclose(result_a.pred_train_fit, result_b.pred_train_fit)
    np.testing.assert_allclose(result_a.pred_val_alpha, result_b.pred_val_alpha)
    np.testing.assert_allclose(result_a.y_train_fit, result_b.y_train_fit)
    np.testing.assert_allclose(result_a.y_val_alpha, result_b.y_val_alpha)
    assert not np.allclose(result_a.y_cert, result_b.y_cert)
