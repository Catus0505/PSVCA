from __future__ import annotations

import pandas as pd


def run_gpu_batch(
    *,
    driver,
    target_groups: dict[int, tuple[int, ...]],
    metadata_for=None,
) -> pd.DataFrame:
    """GPU batch backend entry point for candidate-group certification.

    Phase P0 is a scaffold only: no torch tensor computation is performed here.
    The signature mirrors the driver candidate-group batch entry and falls back
    to the existing CPU oracle path so backend="gpu" is bitwise equivalent to
    backend="cpu". Future phases replace this body with the gather -> batched
    compute -> scatter implementation described in PSVCA_gpu_backend_design.md
    section 3.4.
    """
    return driver._candidate_group_edges_cpu(target_groups, metadata_for=metadata_for)
