from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.linalg.fwl import incremental_r2_direct_ols, incremental_r2_fwl_ols


def test_fwl_incremental_r2_matches_direct_joint_ols() -> None:
    rng = np.random.default_rng(23)
    n = 160
    z = rng.normal(size=(n, 5))
    X_base = np.column_stack(
        [
            z[:, 0],
            z[:, 1],
            0.5 * z[:, 0] + 0.4 * z[:, 2],
            z[:, 3],
        ]
    )
    X_add = np.column_stack(
        [
            0.7 * X_base[:, 0] - 0.2 * X_base[:, 1] + z[:, 4],
            0.3 * X_base[:, 2] + rng.normal(size=n),
        ]
    )
    y = 1.2 + X_base @ np.array([0.7, -0.4, 0.3, 0.0]) + X_add @ np.array([0.9, -0.6])
    y = y + 0.05 * rng.normal(size=n)

    direct = incremental_r2_direct_ols(y, X_base, X_add)
    fwl = incremental_r2_fwl_ols(y, X_base, X_add)
    np.testing.assert_allclose(fwl, direct, atol=1e-10, rtol=1e-10)


def test_fwl_incremental_r2_near_zero_when_add_has_no_unique_signal() -> None:
    rng = np.random.default_rng(29)
    n = 120
    X_base = rng.normal(size=(n, 3))
    X_add = X_base @ np.array([[0.5, -0.2], [1.0, 0.3], [-0.4, 0.7]])
    y = -0.3 + X_base @ np.array([1.0, -0.5, 0.25])

    direct = incremental_r2_direct_ols(y, X_base, X_add)
    fwl = incremental_r2_fwl_ols(y, X_base, X_add)
    assert abs(direct) < 1e-12
    assert abs(fwl) < 1e-12
