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
