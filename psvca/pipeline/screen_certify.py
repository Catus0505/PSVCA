from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from psvca.admission.aggregate import aggregate_certified_edges
from psvca.certify.fdr import FDRConfig, apply_bh_fdr
from psvca.certify.stability import StabilityConfig, apply_stability
from psvca.config import PSVCAConfig
from psvca.io.artifacts import ensure_run_dir, make_run_id


def finalize_screen_certify_edges(
    cfg: PSVCAConfig,
    *,
    tier: str | None = None,
    top_m: int | None = None,
    input_path: str | Path | None = None,
    output_root: str | Path = "runs/phase7_screen_certify",
) -> tuple[pd.DataFrame, dict, Path]:
    effective_cfg = replace(cfg, tier=tier or cfg.tier)
    run_id = make_run_id(effective_cfg)
    source_path = Path(input_path) if input_path is not None else Path("runs") / "phase5_screen_certify" / run_id / "candidate_group_edges.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"screen-certify candidate edges not found: {source_path}")
    edges = pd.read_parquet(source_path)
    if top_m is not None and "screen_rank" in edges.columns:
        edges = edges[edges["screen_rank"] <= int(top_m)].copy()
    fdr_edges = apply_bh_fdr(edges, FDRConfig(q=0.1, min_B_for_formal=200)).edges
    stable = apply_stability(fdr_edges, [fdr_edges], StabilityConfig(min_fraction=1.0)).edges
    aggregate = aggregate_certified_edges(stable).edges
    run_dir = ensure_run_dir(output_root, run_id if top_m is None else f"{run_id}_top{int(top_m)}")
    out_path = run_dir / "edges.parquet"
    aggregate.to_parquet(out_path, index=False)
    summary = {
        "dataset": effective_cfg.dataset,
        "pred_len": int(effective_cfg.pred_len),
        "tier": effective_cfg.tier,
        "top_m": None if top_m is None else int(top_m),
        "n_edges": int(len(aggregate)),
        "n_screen_edges": int(aggregate["passed_screen"].sum()) if "passed_screen" in aggregate.columns else int(len(aggregate)),
        "n_e_certified": int(aggregate["e_certified"].sum()),
        "output": str(out_path),
    }
    (run_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return aggregate, summary, run_dir
