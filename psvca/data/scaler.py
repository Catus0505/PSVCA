from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2D array, got shape {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError("cannot fit scaler on an empty array")
        self.mean_ = arr.mean(axis=0)
        scale = arr.std(axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before transform")
        arr = np.asarray(x, dtype=np.float64)
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
