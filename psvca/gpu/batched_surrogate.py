from __future__ import annotations

import hashlib

import numpy as np

from psvca.gpu.device import resolve_gpu_device


def batched_phase_surrogate(
    x: np.ndarray,
    *,
    source_indices,
    B: int,
    seed: int,
    dtype: str | np.dtype = "float32",
    device: str | None = None,
    assert_cpu_equiv: bool = False,
) -> np.ndarray:
    """Batch phase-randomize source series with source/surrogate-id seeds."""
    import torch

    device = resolve_gpu_device(device=device)

    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"x must have shape (n_source, T), got {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("x must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must be finite")
    if int(B) <= 0:
        raise ValueError("B must be positive")
    source_arr = np.asarray(source_indices, dtype=np.int64)
    if source_arr.ndim == 0:
        source_arr = np.full(arr.shape[0], int(source_arr), dtype=np.int64)
    if source_arr.ndim != 1 or source_arr.size != arr.shape[0]:
        raise ValueError("source_indices must be a scalar or match n_source")
    if np.any(source_arr < 0):
        raise ValueError("source_indices must be non-negative")

    torch_dtype = torch.float64 if np.dtype(dtype) == np.dtype(np.float64) else torch.float32
    values = torch.as_tensor(arr, dtype=torch_dtype, device=device)
    spectrum = torch.fft.rfft(values, dim=-1)
    randomized = spectrum[:, None, :].expand(arr.shape[0], int(B), spectrum.shape[-1]).clone()

    phase_indices = _phase_indices(arr.shape[1], spectrum.shape[-1])
    if phase_indices.size:
        phases = _phase_matrix(
            source_arr,
            int(B),
            int(seed),
            int(phase_indices.size),
        )
        phases_t = torch.as_tensor(phases, dtype=torch_dtype, device=device)
        rotation = torch.exp(1j * phases_t)
        phase_idx_t = torch.as_tensor(phase_indices, dtype=torch.long, device=device)
        randomized.index_copy_(
            2,
            phase_idx_t,
            randomized.index_select(2, phase_idx_t) * rotation,
        )

    out = torch.fft.irfft(randomized, n=arr.shape[1], dim=-1)
    std = torch.std(values, dim=1, correction=0)
    if arr.shape[1] < 3:
        constant = torch.ones_like(std, dtype=torch.bool)
    else:
        constant = std <= torch.finfo(torch_dtype).eps
    if bool(torch.any(constant).detach().cpu()):
        out[constant] = values[constant, None, :]
    result = _tensor_to_numpy(out)
    if assert_cpu_equiv:
        _assert_cpu_equiv(result, arr, source_arr, int(B), int(seed))
    return result


def batched_phase_surrogate_pairs(
    x: np.ndarray,
    *,
    source_indices,
    surrogate_ids,
    seed: int,
    dtype: str | np.dtype = "float32",
    device: str | None = None,
    assert_cpu_equiv: bool = False,
) -> np.ndarray:
    """Phase-randomize one row per (source_idx, surrogate_id) pair."""
    import torch

    device = resolve_gpu_device(device=device)

    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"x must have shape (n_pair, T), got {arr.shape}")
    source_arr = np.asarray(source_indices, dtype=np.int64)
    surrogate_arr = np.asarray(surrogate_ids, dtype=np.int64)
    if source_arr.ndim != 1 or surrogate_arr.ndim != 1:
        raise ValueError("source_indices and surrogate_ids must be 1D")
    if source_arr.size != arr.shape[0] or surrogate_arr.size != arr.shape[0]:
        raise ValueError("source_indices/surrogate_ids must match n_pair")
    if np.any(source_arr < 0) or np.any(surrogate_arr < 0):
        raise ValueError("source_indices and surrogate_ids must be non-negative")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("x must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must be finite")

    torch_dtype = torch.float64 if np.dtype(dtype) == np.dtype(np.float64) else torch.float32
    values = torch.as_tensor(arr, dtype=torch_dtype, device=device)
    spectrum = torch.fft.rfft(values, dim=-1)
    randomized = spectrum.clone()

    phase_indices = _phase_indices(arr.shape[1], spectrum.shape[-1])
    if phase_indices.size:
        phases = _phase_matrix_pairs(
            source_arr,
            surrogate_arr,
            int(seed),
            int(phase_indices.size),
        )
        phases_t = torch.as_tensor(phases, dtype=torch_dtype, device=device)
        rotation = torch.exp(1j * phases_t)
        phase_idx_t = torch.as_tensor(phase_indices, dtype=torch.long, device=device)
        randomized.index_copy_(
            1,
            phase_idx_t,
            randomized.index_select(1, phase_idx_t) * rotation,
        )

    out = torch.fft.irfft(randomized, n=arr.shape[1], dim=-1)
    std = torch.std(values, dim=1, correction=0)
    if arr.shape[1] < 3:
        constant = torch.ones_like(std, dtype=torch.bool)
    else:
        constant = std <= torch.finfo(torch_dtype).eps
    if bool(torch.any(constant).detach().cpu()):
        out[constant] = values[constant]
    result = _tensor_to_numpy(out)
    if assert_cpu_equiv:
        _assert_cpu_equiv_pairs(result, arr, source_arr, surrogate_arr, int(seed))
    return result


def _phase_indices(n_time: int, n_freq: int) -> np.ndarray:
    if int(n_time) % 2 == 0:
        return np.arange(1, int(n_freq) - 1, dtype=np.int64)
    return np.arange(1, int(n_freq), dtype=np.int64)


def _phase_matrix(
    source_indices: np.ndarray,
    B: int,
    seed: int,
    n_phase: int,
) -> np.ndarray:
    phases = np.empty((len(source_indices), int(B), int(n_phase)), dtype=np.float64)
    for source_pos, source_idx in enumerate(source_indices):
        for surrogate_id in range(int(B)):
            rng = np.random.default_rng(
                _derived_seed(seed=seed, source_idx=int(source_idx), surrogate_id=surrogate_id)
            )
            phases[source_pos, surrogate_id] = rng.uniform(0.0, 2.0 * np.pi, size=int(n_phase))
    return phases


def _phase_matrix_pairs(
    source_indices: np.ndarray,
    surrogate_ids: np.ndarray,
    seed: int,
    n_phase: int,
) -> np.ndarray:
    phases = np.empty((len(source_indices), int(n_phase)), dtype=np.float64)
    for pair_pos, (source_idx, surrogate_id) in enumerate(zip(source_indices, surrogate_ids)):
        rng = np.random.default_rng(
            _derived_seed(seed=seed, source_idx=int(source_idx), surrogate_id=int(surrogate_id))
        )
        phases[pair_pos] = rng.uniform(0.0, 2.0 * np.pi, size=int(n_phase))
    return phases


def _derived_seed(*, seed: int, source_idx: int, surrogate_id: int) -> int:
    payload = f"{int(seed)}:{int(source_idx)}:{int(surrogate_id)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _tensor_to_numpy(tensor) -> np.ndarray:
    cpu_tensor = tensor.detach().cpu()
    try:
        return np.asarray(cpu_tensor.numpy())
    except RuntimeError:
        return np.asarray(cpu_tensor.tolist())


def _assert_cpu_equiv(
    result: np.ndarray,
    values: np.ndarray,
    source_indices: np.ndarray,
    B: int,
    seed: int,
) -> None:
    from psvca.nulls.phase_surrogate import make_phase_surrogate

    for source_pos, source_idx in enumerate(source_indices):
        for surrogate_id in range(int(B)):
            expected = make_phase_surrogate(
                values[source_pos],
                source_idx=int(source_idx),
                surrogate_id=int(surrogate_id),
                seed=int(seed),
                dataset="gpu_assert",
                split="pre_test",
                cache_dir=None,
            ).values
            np.testing.assert_allclose(
                result[source_pos, surrogate_id],
                expected,
                rtol=1e-12,
                atol=1e-12,
            )


def _assert_cpu_equiv_pairs(
    result: np.ndarray,
    values: np.ndarray,
    source_indices: np.ndarray,
    surrogate_ids: np.ndarray,
    seed: int,
) -> None:
    from psvca.nulls.phase_surrogate import make_phase_surrogate

    for pair_pos, (source_idx, surrogate_id) in enumerate(zip(source_indices, surrogate_ids)):
        expected = make_phase_surrogate(
            values[pair_pos],
            source_idx=int(source_idx),
            surrogate_id=int(surrogate_id),
            seed=int(seed),
            dataset="gpu_assert",
            split="pre_test",
            cache_dir=None,
        ).values
        np.testing.assert_allclose(
            result[pair_pos],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
