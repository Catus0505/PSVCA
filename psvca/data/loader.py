from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from psvca.config import PSVCAConfig
from psvca.data.registry import DatasetInfo, dataset_path, get_dataset_info
from psvca.data.scaler import StandardScaler
from psvca.data.splits import SplitRanges, carve_pretest_splits, compute_itransformer_borders


@dataclass(frozen=True)
class LoadedSeries:
    values: np.ndarray
    channels: tuple[str, ...]
    splits: SplitRanges
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    source_path: str


def _count_csv_rows(path: Path) -> int:
    with path.open("rb") as f:
        line_count = sum(1 for _ in f)
    if line_count <= 1:
        raise ValueError(f"CSV has no data rows: {path}")
    return line_count - 1


def _make_smoke_raw(n_rows: int = 720, n_channels: int = 4) -> tuple[pd.DataFrame, str]:
    t = np.arange(n_rows, dtype=np.float64)
    data = {
        f"x{k}": np.sin(t / (7.0 + k)) + 0.1 * k * np.cos(t / (17.0 + k))
        for k in range(n_channels)
    }
    return pd.DataFrame(data), "synthetic://smoke"


def _select_itransformer_columns(df_raw: pd.DataFrame, info: DatasetInfo) -> pd.DataFrame:
    if info.dataset_type == "Custom":
        if "date" not in df_raw.columns:
            raise ValueError("Custom dataset must contain a date column")
        if info.target not in df_raw.columns:
            raise ValueError(f"target column {info.target!r} not found")
        cols = list(df_raw.columns)
        cols.remove(info.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [info.target]]

    if info.features in {"M", "MS"}:
        if "date" in df_raw.columns:
            return df_raw[df_raw.columns[1:]]
        return df_raw
    if info.features == "S":
        if info.target not in df_raw.columns:
            raise ValueError(f"target column {info.target!r} not found")
        return df_raw[[info.target]]
    raise ValueError(f"unsupported features policy: {info.features}")


def _read_raw_frame(
    cfg: PSVCAConfig,
    info: DatasetInfo,
    n_rows: int,
    read_until: int,
) -> tuple[pd.DataFrame, str]:
    if info.dataset_type == "synthetic":
        df, source = _make_smoke_raw()
        if len(df) != n_rows:
            raise ValueError("internal smoke n_rows mismatch")
        return df.iloc[:read_until].copy(), source

    path = dataset_path(cfg.data_root, info)
    if not path.exists():
        raise FileNotFoundError(f"dataset CSV not found for {info.name}: {path}")
    return pd.read_csv(path, nrows=read_until), str(path)


def _n_rows_for(cfg: PSVCAConfig, info: DatasetInfo) -> int:
    if info.dataset_type == "synthetic":
        return 720
    path = dataset_path(cfg.data_root, info)
    if not path.exists():
        raise FileNotFoundError(f"dataset CSV not found for {info.name}: {path}")
    return _count_csv_rows(path)


def load_series(cfg: PSVCAConfig, *, include_test: bool = False) -> LoadedSeries:
    info = get_dataset_info(cfg.dataset)
    n_rows = _n_rows_for(cfg, info)
    original_train, original_val, original_test = compute_itransformer_borders(
        info.dataset_type,
        n_rows=n_rows,
        seq_len=cfg.lookback,
        pred_len=cfg.pred_len,
    )
    splits = carve_pretest_splits(
        original_train,
        original_val,
        original_test,
        cfg.split_ratios,
        cfg.stability_blocks,
    )

    read_until = n_rows if include_test else splits.pre_test.end
    df_raw, source_path = _read_raw_frame(cfg, info, n_rows, read_until)
    df_data = _select_itransformer_columns(df_raw, info)
    values_raw = df_data.to_numpy(dtype=np.float64)
    if values_raw.shape[0] < splits.pre_test.end:
        raise ValueError(
            f"loaded too few rows for pre_test: got {values_raw.shape[0]}, "
            f"need {splits.pre_test.end}"
        )
    if info.expected_channels is not None and values_raw.shape[1] != info.expected_channels:
        raise ValueError(
            f"{info.name} expected {info.expected_channels} channels, got {values_raw.shape[1]}"
        )

    scaler = StandardScaler()
    scaler.fit(values_raw[splits.train_fit.start : splits.train_fit.end])
    transformed = scaler.transform(values_raw)
    output_end = n_rows if include_test else splits.pre_test.end
    values = transformed[:output_end]

    return LoadedSeries(
        values=values,
        channels=tuple(str(c) for c in df_data.columns),
        splits=splits,
        scaler_mean=np.array(scaler.mean_, copy=True),
        scaler_scale=np.array(scaler.scale_, copy=True),
        source_path=source_path,
    )
