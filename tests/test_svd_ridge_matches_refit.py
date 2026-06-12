from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.linalg.svd_ridge import fit_ridge_path


def _explicit_centered_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    xc = X - x_mean
    yc = y - y_mean
    if alpha == 0.0:
        coef = np.linalg.pinv(xc) @ yc
    else:
        gram = xc.T @ xc + alpha * np.eye(X.shape[1])
        coef = np.linalg.solve(gram, xc.T @ yc)
    intercept = y_mean - float(x_mean @ coef)
    return coef, intercept


def test_svd_ridge_path_matches_explicit_centered_refit() -> None:
    rng = np.random.default_rng(7)
    n = 80
    z = rng.normal(size=(n, 3))
    X = np.column_stack(
        [
            z[:, 0],
            z[:, 1],
            0.8 * z[:, 0] - 0.2 * z[:, 1] + 0.05 * rng.normal(size=n),
            z[:, 2],
            z[:, 0] + z[:, 2],
        ]
    )
    beta = np.array([1.5, -0.7, 0.3, 0.0, 0.8])
    y = 2.25 + X @ beta + 0.1 * rng.normal(size=n)
    X_eval = rng.normal(size=(13, X.shape[1]))
    alphas = np.array([0.0, 0.01, 0.3, 10.0])

    path = fit_ridge_path(X, y, alphas)
    assert path.coef_path.shape == (len(alphas), X.shape[1])
    assert path.intercepts.shape == (len(alphas),)

    for idx, alpha in enumerate(alphas):
        coef, intercept = _explicit_centered_ridge(X, y, float(alpha))
        pred_expected = X_eval @ coef + intercept
        np.testing.assert_allclose(path.coef_path[idx], coef, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(path.intercepts[idx], intercept, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(path.predict(X_eval, float(alpha)), pred_expected, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(path.predict_index(X_eval, idx), pred_expected, atol=1e-8, rtol=1e-8)


def test_predict_requires_exact_grid_alpha() -> None:
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.float64)
    path = fit_ridge_path(X, y, np.array([0.0, 1.0]))
    try:
        path.predict(X, 0.5)
    except ValueError as exc:
        assert "not in this ridge path" in str(exc)
    else:
        raise AssertionError("predict accepted an alpha outside the grid")
