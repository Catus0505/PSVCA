from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PSVCAConfig:
    data_root: str
    dataset: str
    pred_len: int
    lookback: int
    seed: int
    tier: str
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    stability_blocks: int = 3
    alpha_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    null_method: str = "phase"
    B: int = 20


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_path(path_or_name: str) -> Path:
    candidate = Path(path_or_name)
    if candidate.exists():
        return candidate
    if candidate.suffix:
        rooted = _repo_root() / candidate
        if rooted.exists():
            return rooted
    name = candidate.stem if candidate.suffix else path_or_name
    rooted = _repo_root() / "configs" / f"{name}.yaml"
    if rooted.exists():
        return rooted
    raise FileNotFoundError(f"config not found: {path_or_name}")


def _as_tuple(values: Any, *, field: str) -> tuple[Any, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{field} must be a list or tuple")
    return tuple(values)


def load_config(path_or_name: str) -> PSVCAConfig:
    path = _config_path(path_or_name)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    required = {
        "data_root",
        "dataset",
        "pred_len",
        "lookback",
        "seed",
        "tier",
        "split_ratios",
        "stability_blocks",
        "alpha_grid",
        "null_method",
        "B",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"missing config fields in {path}: {missing}")

    ratios = _as_tuple(raw["split_ratios"], field="split_ratios")
    if len(ratios) != 3:
        raise ValueError("split_ratios must contain exactly three values")

    alpha_grid = _as_tuple(raw["alpha_grid"], field="alpha_grid")
    if not alpha_grid:
        raise ValueError("alpha_grid must not be empty")

    data_root = os.environ.get("PSVCA_DATA_ROOT", raw["data_root"])

    return PSVCAConfig(
        data_root=str(data_root),
        dataset=str(raw["dataset"]),
        pred_len=int(raw["pred_len"]),
        lookback=int(raw["lookback"]),
        seed=int(raw["seed"]),
        tier=str(raw["tier"]),
        split_ratios=tuple(float(x) for x in ratios),
        stability_blocks=int(raw["stability_blocks"]),
        alpha_grid=tuple(float(x) for x in alpha_grid),
        null_method=str(raw["null_method"]),
        B=int(raw["B"]),
    )


def config_hash(cfg: PSVCAConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
