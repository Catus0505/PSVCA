from __future__ import annotations

import argparse
import math
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import replace

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

from psvca.certify.probe import PairwiseProbeConfig
from psvca.certify.probe import normalize_n_jobs
from psvca.certify.fdr import FDRConfig
from psvca.config import load_config
from psvca.data.loader import load_series
from psvca.io.artifacts import ensure_run_dir, make_run_id
from psvca.pipeline.driver import CertificationDriver, workload_summary
from psvca.pipeline.reference import B_HEADROOM_C, MIN_FORMAL_B
from psvca.screen.value_screen import ValueScreenConfig, run_value_screen


BLAS_THREADS_POLICY = {
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
}


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return float("nan"), float("nan")
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    stat = spearmanr(x, y, nan_policy="omit")
    return float(stat.statistic), float(stat.pvalue)


def _effective_B(*, tier: str, requested_B: int, n_channels: int) -> int:
    if tier == "formal":
        per_target_m = int(n_channels) - 1
        b_need = math.ceil(B_HEADROOM_C * per_target_m / FDRConfig().q)
        return int(max(requested_B, MIN_FORMAL_B, b_need))
    if tier == "sanity":
        return int(requested_B)
    raise ValueError(f"unsupported tier: {tier!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", choices=("sanity", "formal"), required=True)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = args.backend or cfg.backend
    loaded = load_series(cfg)
    n_channels = loaded.values.shape[1]
    max_targets = n_channels if args.max_targets is None else min(args.max_targets, n_channels)
    targets = tuple(range(max_targets))
    if not targets:
        raise SystemExit("no targets available")
    top_m = min(args.top_m, max(1, n_channels - 1))
    if args.B is None and args.tier == "sanity":
        requested_B = int(min(cfg.B, 5))
    else:
        requested_B = int(cfg.B if args.B is None else args.B)
    B = _effective_B(tier=args.tier, requested_B=requested_B, n_channels=n_channels)
    effective_cfg = replace(cfg, tier=args.tier, B=B, backend=backend)
    n_jobs = normalize_n_jobs(args.n_jobs)

    screen = run_value_screen(
        values=loaded.values,
        channels=loaded.channels,
        splits=loaded.splits,
        lookback=effective_cfg.lookback,
        horizon=effective_cfg.pred_len,
        alphas=effective_cfg.alpha_grid,
        config=ValueScreenConfig(
            top_m=top_m,
            max_targets=max_targets,
            targets=targets,
            seed=effective_cfg.seed,
            alpha_rule="val_grid",
            n_jobs=n_jobs,
        ),
    )

    probe_cfg = PairwiseProbeConfig(
        alphas=effective_cfg.alpha_grid,
        B=B,
        seed=effective_cfg.seed,
        null_method="phase",
        alpha_rule="val_grid",
        skip_null_on_fail=True,
        delta_floor=0.0,
    )
    driver = CertificationDriver(
        values=loaded.values,
        splits=loaded.splits,
        lookback=effective_cfg.lookback,
        horizon=effective_cfg.pred_len,
        probe_config=probe_cfg,
        seed=effective_cfg.seed,
        dataset=effective_cfg.dataset,
        n_jobs=n_jobs,
        backend=backend,
    )

    passed = screen.edges[screen.edges["passed_screen"]].copy()
    screen_meta = {
        (int(row.target), int(row.source)): {
            "s_screen": float(row.s_screen),
            "screen_rank": int(row.screen_rank),
            "passed_screen": bool(row.passed_screen),
        }
        for row in passed.itertuples(index=False)
    }
    target_groups: dict[int, tuple[int, ...]] = {}
    for target in targets:
        target_screen = passed[passed["target"] == target].sort_values("screen_rank")
        group_sources = tuple(int(s) for s in target_screen["source"].tolist())
        if group_sources:
            target_groups[int(target)] = group_sources

    candidate_parts = []
    certify_start = time.monotonic()
    for done, (target, group_sources) in enumerate(sorted(target_groups.items()), start=1):
        candidate_parts.append(
            driver.candidate_group_edges(
                {target: group_sources},
                metadata_for=lambda target, source: screen_meta[(target, source)],
            )
        )
        elapsed = time.monotonic() - certify_start
        print(
            f"[screen-certify target {done}/{len(target_groups)}] done, elapsed {elapsed:.1f}s",
            flush=True,
        )
    candidate_df = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    candidate_workload = workload_summary(candidate_df, mode="candidate_group", B=probe_cfg.B)
    pairwise_rows = []
    run_pairwise_diagnostic = args.tier == "sanity"
    if run_pairwise_diagnostic:
        pairwise_rows = [
            driver.probe_pairwise_row(
                target=target,
                source=source,
                metadata=screen_meta[(target, source)],
            )
            for target, group_sources in target_groups.items()
            for source in group_sources
        ]
    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values(["target", "source"]).reset_index(drop=True)
    if not candidate_df.empty:
        candidate_df = candidate_df.sort_values(["target", "screen_rank", "source"]).reset_index(
            drop=True
        )
    if pairwise_df.empty:
        spearman_r, spearman_p = float("nan"), float("nan")
    else:
        merged = passed.merge(
            pairwise_df[["target", "source", "delta_true"]],
            on=["target", "source"],
            how="inner",
        )
        spearman_r, spearman_p = _spearman(
            merged["s_screen"].to_numpy(dtype=float),
            merged["delta_true"].to_numpy(dtype=float),
        )

    run_dir = ensure_run_dir(Path("runs") / "phase5_screen_certify", make_run_id(effective_cfg))
    screen_path = run_dir / "screen_edges.parquet"
    candidate_path = run_dir / "candidate_group_edges.parquet"
    summary_path = run_dir / "summary.json"
    screen.edges.to_parquet(screen_path, index=False)
    candidate_df.to_parquet(candidate_path, index=False)

    summary = {
        "dataset": effective_cfg.dataset,
        "pred_len": int(effective_cfg.pred_len),
        "tier": args.tier,
        "backend": backend,
        "effective_B": int(B),
        "requested_B": int(requested_B),
        "B_headroom_C": int(B_HEADROOM_C),
        "B_per_target_m": int(n_channels - 1),
        "fdr_q": float(FDRConfig().q),
        "group_size": candidate_workload["group_size"],
        "group_size_min": candidate_workload["group_size_min"],
        "group_size_max": candidate_workload["group_size_max"],
        "group_size_mean": candidate_workload["group_size_mean"],
        "n_targets_screened": int(screen.summary["n_targets_screened"]),
        "max_targets": int(max_targets),
        "top_m": int(top_m),
        "n_screen_edges": int(screen.summary["n_screen_edges"]),
        "spearman_screen_vs_pairwise_delta": spearman_r,
        "spearman_pvalue": spearman_p,
        "pairwise_diagnostic": bool(run_pairwise_diagnostic),
        "n_candidate_group_edges": int(len(candidate_df)),
        "n_edges": int(candidate_workload["n_edges"]),
        "n_skipped": int(candidate_workload["n_skipped"]),
        "svd_count_total": int(candidate_workload["svd_count_total"]),
        "svd_count_own": int(candidate_workload["svd_count_own"]),
        "svd_count_reduced": int(candidate_workload["svd_count_reduced"]),
        "svd_count_full": int(candidate_workload["svd_count_full"]),
        "svd_count_null": int(candidate_workload["svd_count_null"]),
        "n_certified_candidate_group": (
            int(candidate_df["certified_candidate"].sum()) if not candidate_df.empty else 0
        ),
        "n_jobs_requested": int(args.n_jobs),
        "n_jobs_effective": int(n_jobs),
        "cpu_count": int(os.cpu_count() or 1),
        "blas_threads_policy": BLAS_THREADS_POLICY,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("summary:")
    for key, value in summary.items():
        print(f"  {key}={value}")
    print(f"  screen_edges={screen_path}")
    print(f"  candidate_group_edges={candidate_path}")
    print(f"  summary={summary_path}")


if __name__ == "__main__":
    main()
