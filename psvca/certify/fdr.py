from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FDRConfig:
    q: float = 0.1
    min_B_for_formal: int = 200


@dataclass(frozen=True)
class FDRResult:
    edges: pd.DataFrame
    summary: dict


def _require_columns(edges: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in edges.columns]
    if missing:
        raise ValueError(f"edges missing required columns: {missing}")


def apply_bh_fdr(edges: pd.DataFrame, config: FDRConfig = FDRConfig()) -> FDRResult:
    if not (0.0 < float(config.q) <= 1.0):
        raise ValueError("q must be in (0, 1]")
    if config.min_B_for_formal < 1:
        raise ValueError("min_B_for_formal must be positive")
    _require_columns(edges, ("p_value", "certified_candidate"))

    out = edges.copy()
    p = pd.to_numeric(out["p_value"], errors="coerce").to_numpy(dtype=np.float64)
    candidate = out["certified_candidate"].fillna(False).astype(bool).to_numpy()
    finite_mask = np.isfinite(p)
    raw_reject = np.zeros(len(out), dtype=bool)

    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size:
        finite_p = p[finite_indices]
        order = np.argsort(finite_p, kind="mergesort")
        sorted_p = finite_p[order]
        m = sorted_p.size
        thresholds = (np.arange(1, m + 1, dtype=np.float64) / float(m)) * float(config.q)
        passing = np.flatnonzero(sorted_p <= thresholds)
        if passing.size:
            cutoff = float(sorted_p[int(passing[-1])])
            raw_reject[finite_indices] = finite_p <= cutoff

    if "B" in out.columns:
        b_values = pd.to_numeric(out["B"], errors="coerce")
        underpowered = b_values.fillna(-np.inf).to_numpy(dtype=np.float64) < float(
            config.min_B_for_formal
        )
    else:
        underpowered = np.zeros(len(out), dtype=bool)

    out["fdr_q"] = float(config.q)
    out["fdr_pass"] = raw_reject & candidate
    out["fdr_underpowered"] = underpowered
    summary = {
        "fdr_q": float(config.q),
        "min_B_for_formal": int(config.min_B_for_formal),
        "n_edges": int(len(out)),
        "n_finite_p": int(finite_mask.sum()),
        "n_bh_reject": int(raw_reject.sum()),
        "n_fdr_pass": int(out["fdr_pass"].sum()),
        "fdr_underpowered": bool(np.any(underpowered)),
        "n_fdr_underpowered": int(np.count_nonzero(underpowered)),
    }
    return FDRResult(edges=out, summary=summary)
