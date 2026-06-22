from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from psvca.io.schema import MODEL_INPUT_SCHEMA_VERSION


def _edge_index(value: Any, channels: tuple[str, ...], channel_to_index: dict[str, int], name: str) -> int:
    if isinstance(value, str):
        if value not in channel_to_index:
            raise ValueError(f"{name} channel not found in channels: {value!r}")
        return channel_to_index[value]
    if pd.isna(value):
        raise ValueError(f"{name} contains missing value")
    idx = int(value)
    if idx < 0 or idx >= len(channels):
        raise ValueError(f"{name} index out of bounds: {idx}")
    return idx


def _meta_value(edges_df: pd.DataFrame, column: str, default: Any = None) -> Any:
    if column not in edges_df.columns or edges_df.empty:
        return default
    values = edges_df[column].dropna().unique()
    if len(values) == 0:
        return default
    if len(values) > 1:
        return str(values[0])
    value = values[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_model_input_arrays(
    edges_df: pd.DataFrame,
    channels,
    *,
    weight_col: str = "aligned_gain",
    require_e_certified: bool = True,
) -> dict:
    channels_tuple = tuple(str(c) for c in channels)
    if not channels_tuple:
        raise ValueError("channels must not be empty")
    if "target" not in edges_df.columns or "source" not in edges_df.columns:
        raise ValueError("edges_df must contain target and source")
    if require_e_certified and "e_certified" not in edges_df.columns:
        raise ValueError("edges_df must contain e_certified")

    actual_weight_col = weight_col
    if actual_weight_col not in edges_df.columns:
        if "delta_true" not in edges_df.columns:
            raise ValueError(f"weight column {weight_col!r} missing and delta_true fallback unavailable")
        actual_weight_col = "delta_true"

    n = len(channels_tuple)
    channel_to_index = {channel: idx for idx, channel in enumerate(channels_tuple)}
    A = np.zeros((n, n), dtype=np.bool_)
    W = np.zeros((n, n), dtype=np.float64)
    if "e_certified" in edges_df.columns:
        mask = edges_df["e_certified"].fillna(False).astype(bool)
    else:
        mask = pd.Series([not require_e_certified] * len(edges_df), index=edges_df.index)

    for _, row in edges_df.loc[mask].iterrows():
        target = _edge_index(row["target"], channels_tuple, channel_to_index, "target")
        source = _edge_index(row["source"], channels_tuple, channel_to_index, "source")
        if target == source:
            continue
        weight = row[actual_weight_col]
        if pd.isna(weight):
            weight = 0.0
        A[target, source] = True
        W[target, source] = float(weight)

    np.fill_diagonal(A, False)
    np.fill_diagonal(W, 0.0)
    meta = {
        "dataset": _meta_value(edges_df, "dataset"),
        "pred_len": _meta_value(edges_df, "pred_len"),
        "schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "edge_schema_version": _meta_value(edges_df, "schema_version"),
        "config_hash": _meta_value(edges_df, "config_hash"),
        "git_hash": _meta_value(edges_df, "git_hash"),
        "seed": _meta_value(edges_df, "seed"),
        "tier": _meta_value(edges_df, "tier"),
        "weight_col": actual_weight_col,
        "requested_weight_col": weight_col,
        "n_channels": int(n),
        "n_e_certified": int(A.sum()),
    }
    return {
        "A_certified": A,
        "W_value": W,
        "channels": np.asarray(channels_tuple, dtype=str),
        "meta": meta,
    }


def save_model_input_npz(path, arrays: dict) -> None:
    out_path = Path(path)
    A = np.asarray(arrays["A_certified"])
    W = np.asarray(arrays["W_value"], dtype=np.float64)
    channels = np.asarray(arrays["channels"], dtype=str)
    meta = dict(arrays["meta"])
    n = len(channels)
    if A.shape != (n, n):
        raise ValueError("A_certified shape must be (N, N)")
    if W.shape != (n, n):
        raise ValueError("W_value shape must be (N, N)")
    if A.dtype != np.bool_:
        raise ValueError("A_certified dtype must be bool")
    if np.any(np.diag(A)):
        raise ValueError("A_certified diagonal must be False")
    if np.any(np.diag(W) != 0.0):
        raise ValueError("W_value diagonal must be zero")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        A_certified=A,
        W_value=W,
        channels=channels,
        meta=json.dumps(meta, sort_keys=True),
    )
