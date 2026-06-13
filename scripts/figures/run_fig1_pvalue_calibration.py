from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import kstest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.figures.evidence import (
    ensure_dir,
    null_pair_values,
    ordered_parallel_map,
    probe_value_pair,
    result_record,
    standard_splits,
    write_tsv,
)


def _run_null_pair(task: tuple[int, int, int, int, str]) -> dict:
    pair_id, base_seed, n, B, tier = task
    pair_seed = base_seed + 1000003 * pair_id
    values = null_pair_values(pair_seed, n)
    result = probe_value_pair(
        values,
        target=0,
        source=1,
        splits=standard_splits(n),
        lookback=5,
        horizon=1,
        B=B,
        seed=pair_seed,
        alphas=(0.1, 1.0, 10.0, 100.0),
        dataset=f"fig1_null_{tier}",
    )
    row = result_record(result)
    row.update(tier=tier, seed=base_seed, pair_id=pair_id, B=B)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("smoke", "formal"), required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--n-null", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    n = 260 if args.tier == "smoke" else 900
    tasks = [(pair_id, args.seed, n, args.B, args.tier) for pair_id in range(args.n_null)]
    rows = ordered_parallel_map(_run_null_pair, tasks, args.n_jobs)
    rows = sorted(rows, key=lambda row: int(row["pair_id"]))
    for row in rows:
        print(
            f"pair_id={int(row['pair_id'])} p={float(row['p_value']):.6g} "
            f"candidate={bool(row['certified_candidate'])}"
        )

    out_dir = ensure_dir("runs/figures")
    df = pd.DataFrame(rows)
    detail_path = out_dir / f"fig1_pvalue_calibration_{args.tier}.parquet"
    df.to_parquet(detail_path, index=False)

    p = df["p_value"].to_numpy(dtype=float)
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        raise SystemExit("no finite p-values generated")
    ks = kstest(finite, "uniform")
    summary = [
        {
            "B": args.B,
            "n_null_pairs": len(df),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "empirical_fpr_005": float(np.mean(finite <= 0.05)),
            "mean_p": float(np.mean(finite)),
            "median_p": float(np.median(finite)),
            "candidate_rate_at_005": float(
                np.mean((df["certified_candidate"].to_numpy(dtype=bool)) & (p <= 0.05))
            ),
        }
    ]
    summary_path = out_dir / "fig1_pvalue_calibration_summary.tsv"
    write_tsv(summary_path, summary)
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
