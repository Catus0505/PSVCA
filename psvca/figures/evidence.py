from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import coherence
from scipy.stats import spearmanr

from psvca.certify.probe import PairwiseProbeConfig, PairwiseProbeResult, probe_pairwise
from psvca.linalg.design import DesignMatrix, make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate


@dataclass(frozen=True)
class FigureSplit:
    start: int
    end: int


@dataclass(frozen=True)
class FigureSplits:
    train_fit: FigureSplit
    val_alpha: FigureSplit
    cert: FigureSplit


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def ar1_series(rng: np.random.Generator, n: int, phi: float, scale: float = 1.0) -> np.ndarray:
    x = np.zeros(n, dtype=np.float64)
    noise = rng.normal(scale=scale, size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def standard_splits(n: int) -> FigureSplits:
    train_end = int(0.45 * n)
    val_end = int(0.68 * n)
    cert_end = int(0.95 * n)
    if train_end <= 8 or val_end <= train_end or cert_end <= val_end:
        raise ValueError("n is too small for figure splits")
    return FigureSplits(
        train_fit=FigureSplit(16, train_end),
        val_alpha=FigureSplit(train_end, val_end),
        cert=FigureSplit(val_end, cert_end),
    )


def own_design(values: np.ndarray, target: int, split, lookback: int, horizon: int) -> DesignMatrix:
    return make_lagged_design(
        values,
        target=target,
        sources=(),
        lookback=lookback,
        horizon=horizon,
        y_start=int(split.start),
        y_end=int(split.end),
        include_own=True,
    )


def source_design(
    values: np.ndarray,
    target: int,
    source: int,
    split,
    lookback: int,
    horizon: int,
) -> DesignMatrix:
    return make_lagged_design(
        values,
        target=target,
        sources=(source,),
        lookback=lookback,
        horizon=horizon,
        y_start=int(split.start),
        y_end=int(split.end),
        include_own=False,
    )


def source_level_surrogate_bank(
    values: np.ndarray,
    *,
    target: int,
    source: int,
    splits,
    lookback: int,
    horizon: int,
    B: int,
    seed: int,
    dataset: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    bank: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for surrogate_id in range(B):
        surrogate = make_phase_surrogate(
            values[:, source],
            source_idx=source,
            surrogate_id=surrogate_id,
            seed=seed,
            dataset=dataset,
            split="pre_test",
            cache_dir=None,
        ).values
        s_values = values.copy()
        s_values[:, source] = surrogate
        bank.append(
            (
                source_design(s_values, target, source, splits.train_fit, lookback, horizon).X,
                source_design(s_values, target, source, splits.val_alpha, lookback, horizon).X,
                source_design(s_values, target, source, splits.cert, lookback, horizon).X,
            )
        )
    return bank


def probe_value_pair(
    values: np.ndarray,
    *,
    target: int,
    source: int,
    splits,
    lookback: int,
    horizon: int,
    B: int,
    seed: int,
    alphas: tuple[float, ...],
    dataset: str,
) -> PairwiseProbeResult:
    own_train = own_design(values, target, splits.train_fit, lookback, horizon)
    own_val = own_design(values, target, splits.val_alpha, lookback, horizon)
    own_cert = own_design(values, target, splits.cert, lookback, horizon)
    source_train = source_design(values, target, source, splits.train_fit, lookback, horizon)
    source_val = source_design(values, target, source, splits.val_alpha, lookback, horizon)
    source_cert = source_design(values, target, source, splits.cert, lookback, horizon)
    return probe_pairwise(
        target=target,
        source=source,
        y_train=own_train.y,
        y_val=own_val.y,
        y_cert=own_cert.y,
        own_train=own_train.X,
        own_val=own_val.X,
        own_cert=own_cert.X,
        source_train=source_train.X,
        source_val=source_val.X,
        source_cert=source_cert.X,
        surrogate_bank=source_level_surrogate_bank(
            values,
            target=target,
            source=source,
            splits=splits,
            lookback=lookback,
            horizon=horizon,
            B=B,
            seed=seed,
            dataset=dataset,
        ),
        config=PairwiseProbeConfig(
            alphas=alphas,
            B=B,
            seed=seed,
            null_method="phase",
            alpha_rule="val_grid",
        ),
    )


def result_record(result: PairwiseProbeResult) -> dict:
    row = asdict(result)
    row.pop("delta_null")
    row.pop("alpha_null")
    return row


def write_tsv(path: str | Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def null_pair_values(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = ar1_series(rng, n, phi=0.72)
    source = ar1_series(rng, n, phi=0.68)
    t = np.arange(n, dtype=np.float64)
    target = target + 0.3 * np.sin(2.0 * np.pi * t / 37.0)
    source = source + 0.4 * np.cos(2.0 * np.pi * t / 29.0)
    return np.column_stack([target, source])


def recovery_values(seed: int, n: int, n_channels: int = 10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.column_stack(
        [ar1_series(rng, n, phi=0.45 + 0.03 * (j % 5), scale=0.75) for j in range(n_channels)]
    )
    true_edges = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for target, source in true_edges:
        lag = 3 + (target % 3)
        beta = 0.42 if target % 4 == 0 else -0.38
        for t in range(lag, n):
            values[t, target] += beta * values[t - lag, source]

    t = np.arange(n, dtype=np.float64)
    z1 = 0.8 * np.sin(2.0 * np.pi * t / 41.0) + 0.35 * ar1_series(rng, n, phi=0.7, scale=0.4)
    z2 = 0.7 * np.cos(2.0 * np.pi * t / 53.0) + 0.30 * ar1_series(rng, n, phi=0.65, scale=0.4)
    for target, source, z in ((0, 8, z1), (2, 8, z1), (4, 9, z2), (6, 9, z2)):
        values[:, target] += 0.28 * z + rng.normal(scale=0.08, size=n)
        values[:, source] += 0.32 * z + rng.normal(scale=0.08, size=n)
    return values


def recovery_edge_list() -> list[tuple[int, int, str]]:
    return [
        (0, 1, "TRUE"),
        (2, 3, "TRUE"),
        (4, 5, "TRUE"),
        (6, 7, "TRUE"),
        (1, 3, "NULL"),
        (3, 5, "NULL"),
        (5, 7, "NULL"),
        (7, 1, "NULL"),
        (0, 8, "DECOY"),
        (2, 8, "DECOY"),
        (4, 9, "DECOY"),
        (6, 9, "DECOY"),
    ]


def coherence_features(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    nperseg = min(256, max(32, len(x) // 4))
    _, cxy = coherence(x, y, nperseg=nperseg)
    finite = cxy[np.isfinite(cxy)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.max(finite))


def xcorr_absmax(x: np.ndarray, y: np.ndarray, max_lag: int = 24) -> float:
    x0 = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    y0 = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom <= np.finfo(np.float64).eps:
        return float("nan")
    vals = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            vals.append(float(np.sum(x0[:lag] * y0[-lag:]) / denom))
        elif lag > 0:
            vals.append(float(np.sum(x0[lag:] * y0[:-lag]) / denom))
        else:
            vals.append(float(np.sum(x0 * y0) / denom))
    return float(np.max(np.abs(vals)))


def finite_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def top_high_association_rejected(df: pd.DataFrame, k: int = 3) -> str:
    sub = df[~df["certified_candidate"]].copy()
    if sub.empty:
        return ""
    sub = sub.sort_values("coherence_peak", ascending=False).head(k)
    return "; ".join(
        f"{int(r.target)}<-{int(r.source)}: coherence_peak={r.coherence_peak:.3g}, "
        f"aligned_gain={r.aligned_gain:.3g}, p={r.p_value:.3g}"
        for r in sub.itertuples()
    )


def top_direction_asymmetry(df: pd.DataFrame, k: int = 3) -> str:
    values: dict[tuple[int, int], float] = {}
    for row in df.itertuples():
        values[(int(row.target), int(row.source))] = float(row.aligned_gain)
    pairs = []
    seen: set[tuple[int, int]] = set()
    for i, j in values:
        key = tuple(sorted((i, j)))
        if i == j or key in seen or (j, i) not in values:
            continue
        seen.add(key)
        diff = abs(values[(i, j)] - values[(j, i)])
        pairs.append((diff, i, j, values[(i, j)], values[(j, i)]))
    pairs.sort(reverse=True)
    return "; ".join(
        f"{i}<-{j} vs {j}<-{i}: diff={diff:.3g}, gains={a:.3g}/{b:.3g}"
        for diff, i, j, a, b in pairs[:k]
    )
