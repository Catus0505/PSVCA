from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StabilityConfig:
    min_fraction: float = 0.60
    key_cols: tuple[str, ...] = ("target", "source")


@dataclass(frozen=True)
class StabilityResult:
    edges: pd.DataFrame
    summary: dict


def _require_columns(edges: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in edges.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def apply_stability(
    full_edges: pd.DataFrame,
    block_edges: list[pd.DataFrame],
    config: StabilityConfig = StabilityConfig(),
) -> StabilityResult:
    if not (0.0 <= float(config.min_fraction) <= 1.0):
        raise ValueError("min_fraction must be in [0, 1]")
    if not config.key_cols:
        raise ValueError("key_cols must not be empty")
    required = tuple(config.key_cols) + ("certified_candidate",)
    _require_columns(full_edges, required, "full_edges")

    out = full_edges.copy()
    k = len(block_edges)
    pass_counts = {tuple(row[col] for col in config.key_cols): 0 for _, row in out.iterrows()}
    if k > 0:
        for block_index, block in enumerate(block_edges):
            _require_columns(block, required, f"block_edges[{block_index}]")
            mask = block["certified_candidate"].fillna(False).astype(bool)
            passed = block.loc[mask, list(config.key_cols)].drop_duplicates()
            for _, row in passed.iterrows():
                key = tuple(row[col] for col in config.key_cols)
                if key in pass_counts:
                    pass_counts[key] += 1

    fractions = []
    passes = []
    for _, row in out.iterrows():
        key = tuple(row[col] for col in config.key_cols)
        fraction = 0.0 if k == 0 else pass_counts.get(key, 0) / float(k)
        fractions.append(float(fraction))
        passes.append(bool(k > 0 and fraction >= float(config.min_fraction)))

    out["stability_fraction"] = fractions
    out["stability_pass"] = passes
    out["stability_k"] = int(k)
    out["stability_min_fraction"] = float(config.min_fraction)
    summary = {
        "n_edges": int(len(out)),
        "stability_k": int(k),
        "stability_min_fraction": float(config.min_fraction),
        "n_stability_pass": int(out["stability_pass"].sum()),
        "stability_no_blocks": bool(k == 0),
    }
    if k == 0:
        summary["reason"] = "no block_edges provided"
    return StabilityResult(edges=out, summary=summary)
