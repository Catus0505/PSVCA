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
    *,
    screened_all_edges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _require(exact_edges, ("target", "source", "e_certified"), "exact_edges")
    _require(screened_edges, ("target", "source", "e_certified"), "screened_edges")
    reference_set = _edge_set(exact_edges)
    screen_set = _edge_set(screened_edges)
    intersection = reference_set & screen_set
    all_screened = screened_edges if screened_all_edges is None else screened_all_edges
    _require(all_screened, ("target", "source"), "screened_all_edges")
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
    gap = _gap_breakdown(reference_set, screened_edges, all_screened, config.top_m)
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
        **gap,
    }
    return pd.DataFrame([row])


def _edge_keyed(edges: pd.DataFrame) -> dict[tuple, pd.Series]:
    return {
        (row["target"], row["source"]): row
        for _, row in edges.iterrows()
    }


def _truthy(row: pd.Series, column: str) -> bool:
    if column not in row.index:
        return False
    value = row[column]
    if pd.isna(value):
        return False
    return bool(value)


def _gap_breakdown(
    reference_set: set[tuple],
    screened_edges: pd.DataFrame,
    screened_all_edges: pd.DataFrame,
    top_m: int | None,
) -> dict:
    top_edges = _edge_keyed(screened_edges)
    all_edges = _edge_keyed(screened_all_edges)
    counts = {
        "recovered": 0,
        "lost_at_top_m": 0,
        "lost_not_entering_screen": 0,
        "lost_at_gate": 0,
        "lost_at_fdr": 0,
        "lost_at_stability": 0,
    }
    for edge in reference_set:
        top_row = top_edges.get(edge)
        all_row = all_edges.get(edge)
        if top_row is not None and _truthy(top_row, "e_certified"):
            counts["recovered"] += 1
        elif all_row is None:
            counts["lost_not_entering_screen"] += 1
        elif top_row is None or (
            top_m is not None
            and "screen_rank" in all_row.index
            and not pd.isna(all_row["screen_rank"])
            and int(all_row["screen_rank"]) > int(top_m)
        ):
            counts["lost_at_top_m"] += 1
        elif not _truthy(top_row, "certified_candidate"):
            counts["lost_at_gate"] += 1
        elif "fdr_pass" in top_row.index and not _truthy(top_row, "fdr_pass"):
            counts["lost_at_fdr"] += 1
        elif "stability_pass" in top_row.index and not _truthy(top_row, "stability_pass"):
            counts["lost_at_stability"] += 1
        else:
            counts["lost_at_gate"] += 1

    counts["aggregation_gap"] = counts["lost_at_top_m"] + counts["lost_not_entering_screen"]
    counts["legal_noncertified"] = (
        counts["lost_at_gate"] + counts["lost_at_fdr"] + counts["lost_at_stability"]
    )
    return {key: int(value) for key, value in counts.items()}
