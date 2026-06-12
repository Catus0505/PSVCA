from __future__ import annotations

from typing import Protocol

import numpy as np


class OwnBase(Protocol):
    """Nuisance own-history model used only to define target innovations.

    The own-base maps target own-history windows to target future values, producing
    residuals `r_i = y_i_future - f_own(X_i_past)`. It is a measurement nuisance,
    not the certifier and not a consumer forecasting model. Later certification
    statistics remain independent linear tests, and consumer models such as
    PatchTST or iTransformer are not used as certifiers in this phase.
    """

    def fit(
        self,
        X_past_train_fit: np.ndarray,
        y_future_train_fit: np.ndarray,
        seed: int,
    ) -> None:
        ...

    def predict(self, X_past: np.ndarray) -> np.ndarray:
        ...
