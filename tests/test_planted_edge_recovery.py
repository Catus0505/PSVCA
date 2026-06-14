from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.admission.aggregate import aggregate_certified_edges
from psvca.certify.fdr import FDRConfig, apply_bh_fdr
from psvca.certify.stability import StabilityConfig, apply_stability
from scripts.run_synthetic_check import run_phase6_synthetic


def test_bh_fdr_step_up_candidates_and_underpowered_flag() -> None:
    edges = pd.DataFrame(
        [
            {"target": 0, "source": 1, "p_value": 0.01, "certified_candidate": True, "B": 20},
            {"target": 0, "source": 2, "p_value": 0.02, "certified_candidate": False, "B": 20},
            {"target": 0, "source": 3, "p_value": np.nan, "certified_candidate": True, "B": 20},
            {"target": 1, "source": 0, "p_value": 0.20, "certified_candidate": True, "B": 20},
        ]
    )

    result = apply_bh_fdr(edges, FDRConfig(q=0.1, min_B_for_formal=200))
    out = result.edges.sort_values(["target", "source"]).reset_index(drop=True)

    assert out.loc[0, "fdr_pass"]
    assert not out.loc[1, "fdr_pass"]
    assert not out.loc[2, "fdr_pass"]
    assert not out.loc[3, "fdr_pass"]
    assert bool(out["fdr_underpowered"].all())
    assert result.summary["n_bh_reject"] == 2
    assert result.summary["n_fdr_pass"] == 1


def test_stability_fraction_uses_block_candidate_only() -> None:
    full = pd.DataFrame(
        [
            {"target": 0, "source": 1, "certified_candidate": True, "fdr_pass": True},
            {"target": 0, "source": 2, "certified_candidate": True, "fdr_pass": True},
            {"target": 0, "source": 3, "certified_candidate": True, "fdr_pass": False},
        ]
    )
    blocks = [
        pd.DataFrame(
            [
                {"target": 0, "source": 1, "certified_candidate": True, "fdr_pass": True},
                {"target": 0, "source": 2, "certified_candidate": True, "fdr_pass": False},
            ]
        ),
        pd.DataFrame(
            [
                {"target": 0, "source": 1, "certified_candidate": True, "fdr_pass": True},
                {"target": 0, "source": 3, "certified_candidate": True, "fdr_pass": False},
            ]
        ),
        pd.DataFrame(
            [
                {"target": 0, "source": 2, "certified_candidate": True, "fdr_pass": True},
            ]
        ),
    ]

    # Per-block selection only checks certified_candidate; fdr_pass is intentionally ignored
    # (M-B uses selection frequency, not within-subsample FDR reruns).
    result = apply_stability(full, blocks, StabilityConfig(min_fraction=2 / 3))
    out = result.edges.sort_values(["target", "source"]).reset_index(drop=True)

    assert out.loc[0, "stability_fraction"] == 2 / 3
    assert out.loc[0, "stability_pass"]
    assert out.loc[1, "stability_fraction"] == 2 / 3
    assert out.loc[1, "stability_pass"]
    assert out.loc[2, "stability_fraction"] == 1 / 3
    assert not out.loc[2, "stability_pass"]

    empty = apply_stability(full, [], StabilityConfig(min_fraction=0.1))
    assert not empty.edges["stability_pass"].any()
    assert empty.summary["stability_no_blocks"]


def test_aggregate_e_certified_formula_preserves_diagnostics() -> None:
    edges = pd.DataFrame(
        [
            {
                "target": 0,
                "source": 1,
                "certified_candidate": True,
                "fdr_pass": True,
                "stability_pass": True,
                "delta_true": 0.4,
                "aligned_gain": 0.3,
                "p_value": 0.01,
                "B": 20,
            },
            {
                "target": 0,
                "source": 2,
                "certified_candidate": True,
                "fdr_pass": True,
                "stability_pass": False,
                "delta_true": 0.2,
                "aligned_gain": 0.1,
                "p_value": 0.02,
                "B": 20,
            },
        ]
    )

    result = aggregate_certified_edges(edges)
    out = result.edges.sort_values(["target", "source"]).reset_index(drop=True)

    assert out.loc[0, "e_certified"]
    assert not out.loc[1, "e_certified"]
    assert {"delta_true", "aligned_gain", "p_value", "B"}.issubset(out.columns)
    assert result.summary["n_e_certified"] == 1


def test_planted_strong_edge_recovers_zero_and_period_decoy_rejected() -> None:
    edges, summary = run_phase6_synthetic(
        seed=2026,
        lookback=6,
        horizon=1,
        alphas=(0.01, 0.1, 1.0, 10.0),
        B=39,
        fdr_q=0.1,
        stability_min_fraction=2 / 3,
    )

    strong = edges[edges["edge_type"] == "strong"]
    zero = edges[edges["edge_type"] == "zero"]
    decoy = edges[edges["edge_type"] == "shared_period_decoy"]

    assert summary["strong_edges_recovered"] == summary["strong_edges_total"] == 1
    assert bool(strong["e_certified"].iloc[0])
    assert summary["zero_edges_fdr_pass"] == 0
    assert not bool(zero["fdr_pass"].iloc[0])
    assert summary["decoy_edges_e_certified"] == 0
    assert not bool(decoy["e_certified"].iloc[0])
    assert summary["fdr_underpowered"]
