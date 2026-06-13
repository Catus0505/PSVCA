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
    probe_value_pair,
    result_record,
    standard_splits,
    write_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("smoke", "formal"), required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--n-null", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    n = 260 if args.tier == "smoke" else 900
    lookback = 5
    horizon = 1
    alphas = (0.1, 1.0, 10.0, 100.0)
    splits = standard_splits(n)
    rows = []
    for pair_id in range(args.n_null):
        pair_seed = args.seed + 1009 * pair_id
        values = null_pair_values(pair_seed, n)
        result = probe_value_pair(
            values,
            target=0,
            source=1,
            splits=splits,
            lookback=lookback,
            horizon=horizon,
            B=args.B,
            seed=pair_seed,
            alphas=alphas,
            dataset=f"fig1_null_{args.tier}",
        )
        row = result_record(result)
        row.update(tier=args.tier, seed=args.seed, pair_id=pair_id, B=args.B)
        rows.append(row)
        print(f"pair_id={pair_id} p={result.p_value:.6g} candidate={result.certified_candidate}")

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
