from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.linalg.svd_ridge import fit_ridge_path


def test_gcv_alpha_has_near_best_validation_mse() -> None:
    rng = np.random.default_rng(11)
    n_train = 90
    n_val = 120
    n_features = 45
    latent_train = rng.normal(size=(n_train, 8))
    latent_val = rng.normal(size=(n_val, 8))
    mixing = rng.normal(size=(8, n_features))
    X_train = latent_train @ mixing + 0.15 * rng.normal(size=(n_train, n_features))
    X_val = latent_val @ mixing + 0.15 * rng.normal(size=(n_val, n_features))
    beta = np.zeros(n_features)
    beta[:8] = np.array([1.0, -1.2, 0.8, 0.0, 0.5, -0.4, 0.2, 0.1])
    y_train = -0.75 + X_train @ beta + 1.5 * rng.normal(size=n_train)
    y_val = -0.75 + X_val @ beta + 1.5 * rng.normal(size=n_val)
    alphas = np.logspace(-4, 4, 25)

    path = fit_ridge_path(X_train, y_train, alphas)
    val_mse = np.array(
        [np.mean((y_val - path.predict(X_val, float(alpha))) ** 2) for alpha in alphas]
    )
    gcv_alpha = path.best_gcv_alpha()
    gcv_mse = val_mse[int(np.flatnonzero(alphas == gcv_alpha)[0])]
    best_mse = float(np.min(val_mse))

    assert np.isfinite(path.gcv_scores).all()
    assert gcv_mse <= best_mse * 1.10
