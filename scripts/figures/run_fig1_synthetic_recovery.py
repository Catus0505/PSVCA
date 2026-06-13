from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.figures.evidence import (
    ensure_dir,
    probe_value_pair,
    recovery_edge_list,
    recovery_values,
    result_record,
    standard_splits,
    write_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("smoke", "formal"), required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    n = 360 if args.tier == "smoke" else 1400
    values = recovery_values(args.seed, n=n, n_channels=10)
    splits = standard_splits(n)
    lookback = 6
    horizon = 1
    alphas = (0.1, 1.0, 10.0, 100.0)
    rows = []
    for target, source, edge_type in recovery_edge_list():
        result = probe_value_pair(
            values,
            target=target,
            source=source,
            splits=splits,
            lookback=lookback,
            horizon=horizon,
            B=args.B,
            seed=args.seed + 7919 * target + 104729 * source,
            alphas=alphas,
            dataset=f"fig1_recovery_{args.tier}",
        )
        row = result_record(result)
        row.update(
            tier=args.tier,
            seed=args.seed,
            B=args.B,
            edge_type=edge_type,
            admitted_candidate=bool(result.certified_candidate and result.p_value <= 0.10),
            oracle_incremental_r2=float(result.delta_true),
        )
        rows.append(row)
        print(
            f"{edge_type} {target}<-{source} delta={result.delta_true:.6g} "
            f"aligned={result.aligned_gain:.6g} p={result.p_value:.6g} "
            f"admitted={row['admitted_candidate']}"
        )

    out_dir = ensure_dir("runs/figures")
    df = pd.DataFrame(rows)
    detail_path = out_dir / f"fig1_synthetic_recovery_{args.tier}.parquet"
    df.to_parquet(detail_path, index=False)

    decoy = df[df["edge_type"] == "DECOY"]
    decoy_mean = float(decoy["oracle_incremental_r2"].mean()) if not decoy.empty else float("nan")
    decoy_failed = bool(np.isfinite(decoy_mean) and decoy_mean > 0.01)
    if decoy_failed:
        print(
            "WARNING: decoy mean oracle incremental OOS R2 is "
            f"{decoy_mean:.6g} > 0.01"
        )
    summary = []
    for edge_type, group in df.groupby("edge_type", sort=False):
        summary.append(
            {
                "edge_type": edge_type,
                "n_edges": len(group),
                "admission_rate": float(group["admitted_candidate"].mean()),
                "candidate_rate": float(group["certified_candidate"].mean()),
                "mean_delta_true": float(group["delta_true"].mean()),
                "mean_aligned_gain": float(group["aligned_gain"].mean()),
                "median_p": float(group["p_value"].median()),
                "mean_oracle_incremental_r2": float(group["oracle_incremental_r2"].mean()),
                "decoy_oracle_mean_incremental_r2": decoy_mean,
                "decoy_oracle_failed": decoy_failed,
            }
        )
    summary_path = out_dir / "fig1_synthetic_recovery_summary.tsv"
    write_tsv(summary_path, summary)
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
