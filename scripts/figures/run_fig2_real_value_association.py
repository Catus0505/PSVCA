from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.config import load_config
from psvca.data.loader import load_series
from psvca.figures.evidence import (
    coherence_features,
    ensure_dir,
    finite_spearman,
    ordered_parallel_map,
    probe_value_pair,
    result_record,
    top_direction_asymmetry,
    top_high_association_rejected,
    write_tsv,
    xcorr_absmax,
)


def _run_real_pair(task) -> dict:
    (
        target,
        source,
        task_id,
        values,
        splits,
        lookback,
        pred_len,
        B,
        base_seed,
        alphas,
        dataset,
        tier,
    ) = task
    pre = values[splits.pre_test.start : splits.pre_test.end]
    c_mean, c_peak = coherence_features(pre[:, target], pre[:, source])
    xcorr = xcorr_absmax(pre[:, target], pre[:, source], max_lag=min(48, lookback))
    result = probe_value_pair(
        values,
        target=target,
        source=source,
        splits=splits,
        lookback=lookback,
        horizon=pred_len,
        B=B,
        seed=base_seed + 1000003 * task_id,
        alphas=tuple(alphas),
        dataset=dataset,
    )
    row = result_record(result)
    row.update(
        dataset=dataset,
        pred_len=pred_len,
        tier=tier,
        seed=base_seed,
        B=B,
        coherence_mean=c_mean,
        coherence_peak=c_peak,
        xcorr_absmax=xcorr,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", choices=("smoke", "formal"), required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    loaded = load_series(cfg)
    values = loaded.values
    n_channels = values.shape[1]
    if n_channels > 10:
        raise SystemExit(f"Fig.2 exact pairwise evidence is restricted to n_channels <= 10; got {n_channels}")

    pairs = [(target, source) for target in range(n_channels) for source in range(n_channels) if target != source]
    tasks = [
        (
            target,
            source,
            task_id,
            values,
            loaded.splits,
            cfg.lookback,
            cfg.pred_len,
            args.B,
            args.seed,
            cfg.alpha_grid,
            cfg.dataset,
            args.tier,
        )
        for task_id, (target, source) in enumerate(pairs)
    ]
    rows = ordered_parallel_map(_run_real_pair, tasks, args.n_jobs)
    rows = sorted(rows, key=lambda row: (int(row["target"]), int(row["source"])))
    for row in rows:
        print(
            f"{int(row['target'])}<-{int(row['source'])} "
            f"coherence_peak={float(row['coherence_peak']):.6g} "
            f"aligned={float(row['aligned_gain']):.6g} p={float(row['p_value']):.6g} "
            f"candidate={bool(row['certified_candidate'])}"
        )

    out_dir = ensure_dir("runs/figures")
    df = pd.DataFrame(rows)
    detail_path = out_dir / f"fig2_real_value_association_{cfg.dataset}_pl{cfg.pred_len}_{args.tier}.parquet"
    df.to_parquet(detail_path, index=False)

    n_pairs = len(df)
    n_candidates = int(df["certified_candidate"].sum())
    summary = [
        {
            "dataset": cfg.dataset,
            "pred_len": cfg.pred_len,
            "tier": args.tier,
            "B": args.B,
            "n_ordered_pairs": n_pairs,
            "n_candidates": n_candidates,
            "candidate_sparsity": float(n_candidates / n_pairs) if n_pairs else float("nan"),
            "spearman_coherence_mean_aligned_gain": finite_spearman(
                df["coherence_mean"].to_numpy(float), df["aligned_gain"].to_numpy(float)
            ),
            "spearman_coherence_peak_aligned_gain": finite_spearman(
                df["coherence_peak"].to_numpy(float), df["aligned_gain"].to_numpy(float)
            ),
            "spearman_xcorr_absmax_aligned_gain": finite_spearman(
                df["xcorr_absmax"].to_numpy(float), df["aligned_gain"].to_numpy(float)
            ),
            "top_high_association_rejected_edges": top_high_association_rejected(df),
            "top_direction_asymmetry_pairs": top_direction_asymmetry(df),
        }
    ]
    summary_path = out_dir / "fig2_real_value_association_summary.tsv"
    write_tsv(summary_path, summary)
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
