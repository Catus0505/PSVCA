from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from psvca.linalg.svd_ridge import RidgePath, fit_ridge_path


@dataclass
class RidgeOwnBase:
    alphas: np.ndarray
    alpha_rule: str = "gcv"
    _path: RidgePath | None = field(default=None, init=False, repr=False)
    _alpha: float | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        X_past_train_fit: np.ndarray,
        y_future_train_fit: np.ndarray,
        seed: int = 0,
    ) -> None:
        del seed
        if self.alpha_rule != "gcv":
            raise ValueError(f"unsupported alpha_rule for Phase 2: {self.alpha_rule}")
        path = fit_ridge_path(X_past_train_fit, y_future_train_fit, self.alphas)
        alpha = path.best_gcv_alpha()
        self._path = path
        self._alpha = alpha

    def predict(self, X_past: np.ndarray) -> np.ndarray:
        if self._path is None or self._alpha is None:
            raise RuntimeError("RidgeOwnBase must be fit before predict")
        return self._path.predict(X_past, self._alpha)

    @property
    def alpha_(self) -> float:
        if self._alpha is None:
            raise RuntimeError("RidgeOwnBase has not been fit")
        return self._alpha

    @property
    def path_(self) -> RidgePath:
        if self._path is None:
            raise RuntimeError("RidgeOwnBase has not been fit")
        return self._path
