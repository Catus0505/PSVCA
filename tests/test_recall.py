from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.admission.recall import RecallConfig, recall_vs_reduction


def test_recall_vs_reduction_counts_certified_overlap() -> None:
    exact = pd.DataFrame(
        [
            {"target": 0, "source": 1, "e_certified": True},
            {"target": 0, "source": 2, "e_certified": True},
            {"target": 1, "source": 2, "e_certified": False},
            {"target": 2, "source": 0, "e_certified": False},
        ]
    )
    screened = pd.DataFrame(
        [
            {"target": 0, "source": 1, "e_certified": True, "passed_screen": True},
            {"target": 0, "source": 2, "e_certified": False, "passed_screen": True},
        ]
    )

    out = recall_vs_reduction(
        exact,
        screened,
        RecallConfig(dataset="synthetic", pred_len=1, tier="sanity", top_m=2),
    )
    row = out.iloc[0]

    assert row["n_reference_e_certified"] == 2
    assert row["n_screen_e_certified"] == 1
    assert row["n_recalled_e_certified"] == 1
    assert row["recall"] == 0.5
    assert row["reduction"] == 0.5
    assert row["recovered"] == 1
    assert row["lost_at_gate"] == 1
    assert row["aggregation_gap"] == 0


def test_recall_gap_breakdown_separates_screening_from_certification_losses() -> None:
    exact = pd.DataFrame(
        [
            {"target": 0, "source": 1, "e_certified": True},
            {"target": 0, "source": 2, "e_certified": True},
            {"target": 0, "source": 3, "e_certified": True},
            {"target": 1, "source": 0, "e_certified": True},
            {"target": 1, "source": 2, "e_certified": True},
            {"target": 2, "source": 0, "e_certified": True},
        ]
    )
    all_screened = pd.DataFrame(
        [
            {
                "target": 0,
                "source": 1,
                "screen_rank": 1,
                "certified_candidate": True,
                "fdr_pass": True,
                "stability_pass": True,
            },
            {
                "target": 0,
                "source": 2,
                "screen_rank": 3,
                "certified_candidate": True,
                "fdr_pass": True,
                "stability_pass": True,
            },
            {
                "target": 0,
                "source": 3,
                "screen_rank": 2,
                "certified_candidate": False,
                "fdr_pass": False,
                "stability_pass": True,
            },
            {
                "target": 1,
                "source": 0,
                "screen_rank": 1,
                "certified_candidate": True,
                "fdr_pass": False,
                "stability_pass": True,
            },
            {
                "target": 1,
                "source": 2,
                "screen_rank": 2,
                "certified_candidate": True,
                "fdr_pass": True,
                "stability_pass": False,
            },
        ]
    )
    top_m_edges = all_screened[all_screened["screen_rank"] <= 2].copy()
    top_m_edges["e_certified"] = (
        top_m_edges["certified_candidate"]
        & top_m_edges["fdr_pass"]
        & top_m_edges["stability_pass"]
    )

    out = recall_vs_reduction(
        exact,
        top_m_edges,
        RecallConfig(dataset="synthetic", pred_len=1, tier="sanity", top_m=2),
        screened_all_edges=all_screened,
    )
    row = out.iloc[0]

    assert row["recovered"] == 1
    assert row["lost_at_top_m"] == 1
    assert row["lost_not_entering_screen"] == 1
    assert row["lost_at_gate"] == 1
    assert row["lost_at_fdr"] == 1
    assert row["lost_at_stability"] == 1
    assert row["aggregation_gap"] == 2
    assert row["legal_noncertified"] == 3
