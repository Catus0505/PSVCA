from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.certify.probe import PairwiseProbeConfig, normalize_n_jobs
from psvca.config import load_config
from psvca.data.loader import load_series
from psvca.io.artifacts import ensure_run_dir, make_run_id
from psvca.pipeline.reference import pairwise_edge_task, run_reference_pipeline


BLAS_THREADS_POLICY = {
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
}


def _run_pairwise_edge_task(task: dict) -> dict:
    return pairwise_edge_task(task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.tier != "sanity":
        edges, summary, _ = run_reference_pipeline(cfg, tier=args.tier, n_jobs=args.n_jobs)
        print("summary:")
        for key, value in summary.items():
            print(f"  {key}={value}")
        print(f"  n_e_certified={int(edges['e_certified'].sum())}")
        return
    loaded = load_series(cfg)
    n_jobs = normalize_n_jobs(args.n_jobs)
    n_channels = loaded.values.shape[1]
    if n_channels > 10:
        raise SystemExit(
            f"Phase 4 pairwise reference is restricted to small N sanity data; got N={n_channels}"
        )

    probe_cfg = PairwiseProbeConfig(
        alphas=cfg.alpha_grid,
        B=cfg.B,
        seed=cfg.seed,
        null_method="phase",
        alpha_rule="val_grid",
    )
    tasks = [
        {
            "values": loaded.values,
            "splits": loaded.splits,
            "cfg": cfg,
            "probe_cfg": probe_cfg,
            "target": target,
            "source": source,
        }
        for target in range(n_channels)
        for source in range(n_channels)
        if source != target
    ]
    if n_jobs == 1 or len(tasks) <= 1:
        rows = [_run_pairwise_edge_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks))) as executor:
            rows = list(executor.map(_run_pairwise_edge_task, tasks))
    rows.sort(key=lambda row: (row["target"], row["source"]))
    for row in rows:
        print(
            f"target={row['target']} source={row['source']} delta={row['delta_true']:.6g} "
            f"aligned={row['aligned_gain']:.6g} p={row['p_value']:.6g} "
            f"candidate={row['certified_candidate']}"
        )

    df = pd.DataFrame(rows)
    run_dir = ensure_run_dir(Path("runs") / "phase4_reference", make_run_id(cfg))
    out_path = run_dir / "edges_pairwise.parquet"
    df.to_parquet(out_path, index=False)

    p = df["p_value"].to_numpy(dtype=float)
    print("summary:")
    print(f"  n_edges={len(df)}")
    print(f"  n_certified_candidate={int(df['certified_candidate'].sum())}")
    print(f"  p_value_min={np.nanmin(p):.6g}")
    print(f"  p_value_median={np.nanmedian(p):.6g}")
    print(f"  p_value_max={np.nanmax(p):.6g}")
    print(f"  near_zero_target_variance={int(df['near_zero_target_variance'].sum())}")
    print(f"  sparse_zero={int(df['sparse_zero'].sum())}")
    print(f"  unstable_metric={int(df['unstable_metric'].sum())}")
    print(f"  positive_delta={int(df['gate_delta_true'].sum())}")
    print(f"  positive_aligned_gain={int(df['gate_aligned_gain'].sum())}")
    print(f"  n_jobs_requested={int(args.n_jobs)}")
    print(f"  n_jobs_effective={int(n_jobs)}")
    print(f"  cpu_count={int(os.cpu_count() or 1)}")
    print(f"  blas_threads_policy={BLAS_THREADS_POLICY}")
    print(f"  output={out_path}")


if __name__ == "__main__":
    main()
