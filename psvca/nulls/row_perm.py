from __future__ import annotations

import hashlib

import numpy as np


def _stable_seed(seed: int) -> int:
    digest = hashlib.blake2b(str(int(seed)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def row_permute_1d(x: np.ndarray, *, seed: int) -> np.ndarray:
    """Legacy/debug row permutation null.

    This preserves the marginal value multiset but destroys autocorrelation. It is
    not the formal null for PSVCA admission; the formal null is phase surrogate.
    """

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("x must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must be finite")
    rng = np.random.default_rng(_stable_seed(seed))
    return rng.permutation(arr)
