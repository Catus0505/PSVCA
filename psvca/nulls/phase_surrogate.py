from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SurrogateKey:
    dataset: str
    split: str
    source_idx: int
    surrogate_id: int
    seed: int
    n: int


@dataclass(frozen=True)
class SurrogateResult:
    key: SurrogateKey
    values: np.ndarray
    cache_path: Path | None
    cache_hit: bool


def _validate_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("x must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must be finite")
    return arr


def _derived_seed(seed: int, source_idx: int, surrogate_id: int) -> int:
    payload = f"{int(seed)}:{int(source_idx)}:{int(surrogate_id)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _safe_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe.strip("._") or "unknown"


def _cache_path(cache_dir: str | Path, key: SurrogateKey) -> Path:
    name = (
        f"{_safe_part(key.dataset)}__{_safe_part(key.split)}__"
        f"src{key.source_idx}__sid{key.surrogate_id}__seed{key.seed}__n{key.n}.npy"
    )
    return Path(cache_dir) / name


def phase_randomize_1d(x: np.ndarray, *, seed: int) -> np.ndarray:
    arr = _validate_1d(x)
    if arr.size < 3 or float(np.std(arr)) <= np.finfo(np.float64).eps:
        # A constant or nearly constant series has no usable random phase
        # structure to break; returning a copy preserves its spectrum exactly.
        return arr.copy()

    spectrum = np.fft.rfft(arr)
    randomized = spectrum.copy()
    rng = np.random.default_rng(seed)
    n_freq = len(spectrum)
    if arr.size % 2 == 0:
        phase_indices = np.arange(1, n_freq - 1)
    else:
        phase_indices = np.arange(1, n_freq)
    if phase_indices.size:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=phase_indices.size)
        randomized[phase_indices] *= np.exp(1j * phases)

    out = np.fft.irfft(randomized, n=arr.size)
    return np.asarray(out, dtype=np.float64)


def make_phase_surrogate(
    x: np.ndarray,
    *,
    source_idx: int,
    surrogate_id: int,
    seed: int,
    dataset: str = "unknown",
    split: str = "pre_test",
    cache_dir: str | Path | None = None,
) -> SurrogateResult:
    arr = _validate_1d(x)
    if source_idx < 0:
        raise ValueError("source_idx must be non-negative")
    if surrogate_id < 0:
        raise ValueError("surrogate_id must be non-negative")
    key = SurrogateKey(
        dataset=str(dataset),
        split=str(split),
        source_idx=int(source_idx),
        surrogate_id=int(surrogate_id),
        seed=int(seed),
        n=int(arr.size),
    )

    path = _cache_path(cache_dir, key) if cache_dir is not None else None
    if path is not None and path.exists():
        values = np.load(path)
        cached = _validate_1d(values)
        if cached.shape != arr.shape:
            raise ValueError(f"cached surrogate shape mismatch: {path}")
        return SurrogateResult(key=key, values=cached.copy(), cache_path=path, cache_hit=True)

    values = phase_randomize_1d(
        arr,
        seed=_derived_seed(seed=key.seed, source_idx=key.source_idx, surrogate_id=key.surrogate_id),
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values)
    return SurrogateResult(key=key, values=values, cache_path=path, cache_hit=False)


def make_surrogate_bank_for_source(
    x: np.ndarray,
    *,
    source_idx: int,
    B: int,
    seed: int,
    dataset: str = "unknown",
    split: str = "pre_test",
    cache_dir: str | Path | None = None,
) -> list[SurrogateResult]:
    if B <= 0:
        raise ValueError("B must be positive")
    return [
        make_phase_surrogate(
            x,
            source_idx=source_idx,
            surrogate_id=surrogate_id,
            seed=seed,
            dataset=dataset,
            split=split,
            cache_dir=cache_dir,
        )
        for surrogate_id in range(B)
    ]
