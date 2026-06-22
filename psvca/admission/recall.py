from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecallConfig:
    dataset: str
    pred_len: int
    tier: str
    top_m: int | None = None
    candidate_budget: int | None = None
    screen_fraction: float | None = None


def _require(edges: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [col for col in columns if col not in edges.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _edge_set(edges: pd.DataFrame) -> set[tuple]:
    _require(edges, ("target", "source", "e_certified"), "edges")
    mask = edges["e_certified"].fillna(False).astype(bool)
    return set(edges.loc[mask, ["target", "source"]].itertuples(index=False, name=None))


def recall_vs_reduction(
    exact_edges: pd.DataFrame,
    screened_edges: pd.DataFrame,
    config: RecallConfig,
) -> pd.DataFrame:
    _require(exact_edges, ("target", "source", "e_certified"), "exact_edges")
    _require(screened_edges, ("target", "source", "e_certified"), "screened_edges")
    reference_set = _edge_set(exact_edges)
    screen_set = _edge_set(screened_edges)
    intersection = reference_set & screen_set
    reference_empty = len(reference_set) == 0
    if reference_empty:
        recall = float("nan")
    else:
        recall = len(intersection) / float(len(reference_set))

    if "passed_screen" in screened_edges.columns:
        n_screen_edges = int(screened_edges["passed_screen"].fillna(False).astype(bool).sum())
    elif "screen_rank" in screened_edges.columns:
        n_screen_edges = int(len(screened_edges))
    else:
        n_screen_edges = int(len(screened_edges))
    n_exact_edges = int(len(exact_edges))
    reduction = float("nan") if n_exact_edges == 0 else 1.0 - n_screen_edges / float(n_exact_edges)
    row = {
        "dataset": config.dataset,
        "pred_len": int(config.pred_len),
        "tier": config.tier,
        "top_m": config.top_m,
        "candidate_budget": config.candidate_budget,
        "screen_fraction": config.screen_fraction,
        "n_exact_edges": n_exact_edges,
        "n_screen_edges": n_screen_edges,
        "n_reference_e_certified": int(len(reference_set)),
        "n_screen_e_certified": int(len(screen_set)),
        "n_recalled_e_certified": int(len(intersection)),
        "recall": recall,
        "reduction": reduction,
        "reference_empty": bool(reference_empty),
    }
    return pd.DataFrame([row])
