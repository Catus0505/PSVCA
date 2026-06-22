from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.admission.recall import RecallConfig, recall_vs_reduction
from psvca.certify.probe import normalize_n_jobs
from psvca.config import load_config
from psvca.io.artifacts import ensure_run_dir, make_run_id
from psvca.pipeline.screen_certify import finalize_screen_certify_edges


def _parse_top_ms(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--top-ms must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("--top-ms values must be positive")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--top-ms", default="2,4,8")
    parser.add_argument("--reference-edges-path", default=None)
    parser.add_argument("--screen-edges-path", default=None)
    args = parser.parse_args()
    normalize_n_jobs(args.n_jobs)
    cfg = replace(load_config(args.config), tier=args.tier)
    run_id = make_run_id(cfg)
    reference_path = (
        Path(args.reference_edges_path)
        if args.reference_edges_path
        else Path("runs") / "phase7_reference" / run_id / "edges.parquet"
    )
    if not reference_path.exists():
        raise SystemExit(
            f"exact reference edges not found: {reference_path}. "
            "Run scripts/run_reference.py --tier formal first."
        )
    exact_edges = pd.read_parquet(reference_path)
    screen_path = (
        Path(args.screen_edges_path)
        if args.screen_edges_path
        else Path("runs") / "phase5_screen_certify" / run_id / "candidate_group_edges.parquet"
    )
    if not screen_path.exists():
        raise SystemExit(
            f"screen candidate edges not found: {screen_path}. "
            "Run scripts/run_screen_certify.py first."
        )
    all_screen_edges = pd.read_parquet(screen_path)
    rows = []
    for top_m in _parse_top_ms(args.top_ms):
        screened_edges, _, _ = finalize_screen_certify_edges(
            cfg,
            tier=args.tier,
            top_m=top_m,
            input_path=screen_path,
        )
        summary = recall_vs_reduction(
            exact_edges,
            screened_edges,
            RecallConfig(
                dataset=cfg.dataset,
                pred_len=cfg.pred_len,
                tier=args.tier,
                top_m=top_m,
            ),
            screened_all_edges=all_screen_edges,
        )
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True)
    run_dir = ensure_run_dir(Path("runs") / "phase7_recall", run_id)
    csv_path = run_dir / "recall_vs_reduction.csv"
    parquet_path = run_dir / "recall_vs_reduction.parquet"
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)
    print("summary:")
    print(f"  reference_edges={reference_path}")
    print(f"  screen_edges={screen_path}")
    print(f"  n_reference_e_certified={int(exact_edges['e_certified'].fillna(False).astype(bool).sum())}")
    for _, row in out.iterrows():
        print(
            f"  top_m={row['top_m']} recall={row['recall']} "
            f"reduction={row['reduction']} n_screen_e_certified={row['n_screen_e_certified']} "
            f"aggregation_gap={row['aggregation_gap']} legal_noncertified={row['legal_noncertified']}"
        )
    print(f"  recall_csv={csv_path}")
    print(f"  recall_parquet={parquet_path}")


if __name__ == "__main__":
    main()
