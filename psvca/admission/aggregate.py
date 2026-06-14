from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AggregateConfig:
    require_fdr: bool = True
    require_stability: bool = True


@dataclass(frozen=True)
class AggregateResult:
    edges: pd.DataFrame
    summary: dict


def _require_columns(edges: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in edges.columns]
    if missing:
        raise ValueError(f"edges missing required columns: {missing}")


def aggregate_certified_edges(
    edges: pd.DataFrame,
    config: AggregateConfig = AggregateConfig(),
) -> AggregateResult:
    required = ["certified_candidate"]
    if config.require_fdr:
        required.append("fdr_pass")
    if config.require_stability:
        required.append("stability_pass")
    _require_columns(edges, tuple(required))

    out = edges.copy()
    certified = out["certified_candidate"].fillna(False).astype(bool)
    fdr = out["fdr_pass"].fillna(False).astype(bool) if config.require_fdr else True
    stability = (
        out["stability_pass"].fillna(False).astype(bool) if config.require_stability else True
    )
    out["e_certified"] = certified & fdr & stability
    summary = {
        "n_edges": int(len(out)),
        "n_certified_candidate": int(certified.sum()),
        "n_fdr_pass": int(out["fdr_pass"].fillna(False).astype(bool).sum())
        if "fdr_pass" in out.columns
        else 0,
        "n_stability_pass": int(out["stability_pass"].fillna(False).astype(bool).sum())
        if "stability_pass" in out.columns
        else 0,
        "n_e_certified": int(out["e_certified"].sum()),
    }
    return AggregateResult(edges=out, summary=summary)
