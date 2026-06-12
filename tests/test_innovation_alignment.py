from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.base.innovation import compute_innovation_for_target
from psvca.linalg.design import make_lagged_design


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


def test_innovation_indices_and_horizon_alignment() -> None:
    t = np.arange(30, dtype=np.float64)
    values = np.column_stack([t, 100.0 + 2.0 * t])
    splits = SplitsLike(
        train_fit=RangeLike(6, 14),
        val_alpha=RangeLike(14, 20),
        cert=RangeLike(20, 26),
        original_test=RangeLike(26, 30),
    )
    lookback = 3
    horizon = 2

    result = compute_innovation_for_target(
        values=values,
        channels=("x0", "x1"),
        splits=splits,
        target=0,
        lookback=lookback,
        horizon=horizon,
        alphas=np.array([0.0, 0.1, 1.0]),
        seed=0,
    )

    assert np.all((result.train_fit_indices >= 6) & (result.train_fit_indices < 14))
    assert np.all((result.val_alpha_indices >= 14) & (result.val_alpha_indices < 20))
    assert np.all((result.cert_indices >= 20) & (result.cert_indices < 26))
    assert result.cert_indices.max() < splits.original_test.start
    assert len(result.resid_train_fit) == len(result.y_train_fit) == len(result.pred_train_fit)
    assert len(result.resid_val_alpha) == len(result.y_val_alpha) == len(result.pred_val_alpha)
    assert len(result.resid_cert) == len(result.y_cert) == len(result.pred_cert)

    design = make_lagged_design(
        values,
        target=0,
        sources=(),
        lookback=lookback,
        horizon=horizon,
        y_start=6,
        y_end=14,
        include_own=True,
    )
    u = int(design.future_indices[0])
    origin = u - horizon + 1
    expected_window = np.array([origin - lookback, origin - lookback + 1, origin - lookback + 2])
    np.testing.assert_array_equal(design.X[0], expected_window)
    assert result.train_fit_indices[0] == u


def test_innovation_rejects_target_out_of_bounds() -> None:
    values = np.ones((20, 2), dtype=np.float64)
    splits = SplitsLike(
        train_fit=RangeLike(5, 10),
        val_alpha=RangeLike(10, 15),
        cert=RangeLike(15, 18),
        original_test=RangeLike(18, 20),
    )
    try:
        compute_innovation_for_target(
            values,
            ("a", "b"),
            splits,
            target=2,
            lookback=2,
            horizon=1,
            alphas=np.array([1.0]),
        )
    except ValueError as exc:
        assert "target index out of bounds" in str(exc)
    else:
        raise AssertionError("target out of bounds was accepted")
