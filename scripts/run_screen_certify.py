from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
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

from psvca.certify.probe import PairwiseProbeConfig, probe_candidate_group, probe_pairwise
from psvca.certify.probe import normalize_n_jobs
from psvca.config import load_config
from psvca.data.loader import load_series
from psvca.io.artifacts import ensure_run_dir, make_run_id
from psvca.linalg.design import make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate
from psvca.screen.value_screen import ValueScreenConfig, run_value_screen


BLAS_THREADS_POLICY = {
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
}


def _bounds(split) -> tuple[int, int]:
    return int(split.start), int(split.end)


def _own_design(values: np.ndarray, target: int, split, lookback: int, horizon: int):
    start, end = _bounds(split)
    return make_lagged_design(
        values,
        target=target,
        sources=(),
        lookback=lookback,
        horizon=horizon,
        y_start=start,
        y_end=end,
        include_own=True,
    )


def _source_design(values: np.ndarray, target: int, source: int, split, lookback: int, horizon: int):
    start, end = _bounds(split)
    return make_lagged_design(
        values,
        target=target,
        sources=(source,),
        lookback=lookback,
        horizon=horizon,
        y_start=start,
        y_end=end,
        include_own=False,
    )


def _surrogate_bank(
    values: np.ndarray,
    *,
    target: int,
    source: int,
    splits,
    lookback: int,
    horizon: int,
    B: int,
    seed: int,
    dataset: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    bank = []
    for surrogate_id in range(B):
        surrogate = make_phase_surrogate(
            values[:, source],
            source_idx=source,
            surrogate_id=surrogate_id,
            seed=seed,
            dataset=dataset,
            split="pre_test",
            cache_dir=None,
        ).values
        s_values = values.copy()
        s_values[:, source] = surrogate
        bank.append(
            (
                _source_design(s_values, target, source, splits.train_fit, lookback, horizon).X,
                _source_design(s_values, target, source, splits.val_alpha, lookback, horizon).X,
                _source_design(s_values, target, source, splits.cert, lookback, horizon).X,
            )
        )
    return bank


def _pairwise_row(result) -> dict:
    data = asdict(result)
    data.pop("delta_null")
    data.pop("alpha_null")
    data["mode"] = "pairwise"
    data["alpha_rule"] = "val_grid"
    return data


def _candidate_group_row(result, screen_row: pd.Series) -> dict:
    data = asdict(result)
    data.pop("delta_null")
    data.pop("alpha_null")
    data["s_screen"] = float(screen_row["s_screen"])
    data["screen_rank"] = int(screen_row["screen_rank"])
    return data


def _candidate_group_row_from_values(
    result,
    *,
    s_screen: float,
    screen_rank: int,
) -> dict:
    data = asdict(result)
    data.pop("delta_null")
    data.pop("alpha_null")
    data["s_screen"] = float(s_screen)
    data["screen_rank"] = int(screen_rank)
    return data


def _run_candidate_edge_task(task: dict) -> tuple[dict, dict]:
    target = int(task["target"])
    source = int(task["source"])
    cfg = task["cfg"]
    probe_cfg = task["probe_cfg"]
    splits = task["splits"]
    values = task["values"]
    own_train, own_val, own_cert = task["own_designs"]
    source_train_by_source = task["source_train_by_source"]
    source_val_by_source = task["source_val_by_source"]
    source_cert_by_source = task["source_cert_by_source"]
    bank = _surrogate_bank(
        values,
        target=target,
        source=source,
        splits=splits,
        lookback=cfg.lookback,
        horizon=cfg.pred_len,
        B=probe_cfg.B,
        seed=cfg.seed,
        dataset=cfg.dataset,
    )
    pairwise = probe_pairwise(
        target=target,
        source=source,
        y_train=own_train.y,
        y_val=own_val.y,
        y_cert=own_cert.y,
        own_train=own_train.X,
        own_val=own_val.X,
        own_cert=own_cert.X,
        source_train=source_train_by_source[source],
        source_val=source_val_by_source[source],
        source_cert=source_cert_by_source[source],
        surrogate_bank=bank,
        config=probe_cfg,
    )
    candidate = probe_candidate_group(
        target=target,
        source=source,
        group_sources=task["group_sources"],
        y_train=own_train.y,
        y_val=own_val.y,
        y_cert=own_cert.y,
        own_train=own_train.X,
        own_val=own_val.X,
        own_cert=own_cert.X,
        source_train_by_source=source_train_by_source,
        source_val_by_source=source_val_by_source,
        source_cert_by_source=source_cert_by_source,
        surrogate_bank=bank,
        group_id=task["group_id"],
        n_jobs=task["n_jobs"],
        config=probe_cfg,
    )
    return (
        _pairwise_row(pairwise),
        _candidate_group_row_from_values(
            candidate,
            s_screen=float(task["s_screen"]),
            screen_rank=int(task["screen_rank"]),
        ),
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return float("nan"), float("nan")
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    stat = spearmanr(x, y, nan_policy="omit")
    return float(stat.statistic), float(stat.pvalue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    if args.tier != "sanity":
        raise SystemExit("Phase 5 screen certify only supports --tier sanity")
    cfg = load_config(args.config)
    loaded = load_series(cfg)
    n_channels = loaded.values.shape[1]
    targets = tuple(range(min(args.max_targets, n_channels)))
    if not targets:
        raise SystemExit("no targets available")
    top_m = min(args.top_m, max(1, n_channels - 1))
    B = int(min(cfg.B, 5) if args.B is None else args.B)
    n_jobs = normalize_n_jobs(args.n_jobs)

    screen = run_value_screen(
        values=loaded.values,
        channels=loaded.channels,
        splits=loaded.splits,
        lookback=cfg.lookback,
        horizon=cfg.pred_len,
        alphas=cfg.alpha_grid,
        config=ValueScreenConfig(
            top_m=top_m,
            max_targets=args.max_targets,
            targets=targets,
            seed=cfg.seed,
            alpha_rule="val_grid",
            n_jobs=n_jobs,
        ),
    )

    probe_cfg = PairwiseProbeConfig(
        alphas=cfg.alpha_grid,
        B=B,
        seed=cfg.seed,
        null_method="phase",
        alpha_rule="val_grid",
    )
    own_by_target = {
        target: (
            _own_design(loaded.values, target, loaded.splits.train_fit, cfg.lookback, cfg.pred_len),
            _own_design(loaded.values, target, loaded.splits.val_alpha, cfg.lookback, cfg.pred_len),
            _own_design(loaded.values, target, loaded.splits.cert, cfg.lookback, cfg.pred_len),
        )
        for target in targets
    }

    tasks = []
    passed = screen.edges[screen.edges["passed_screen"]].copy()
    for target in targets:
        target_screen = passed[passed["target"] == target].sort_values("screen_rank")
        group_sources = tuple(int(s) for s in target_screen["source"].tolist())
        if not group_sources:
            continue

        source_train_by_source = {}
        source_val_by_source = {}
        source_cert_by_source = {}
        for source in group_sources:
            source_train_by_source[source] = _source_design(
                loaded.values, target, source, loaded.splits.train_fit, cfg.lookback, cfg.pred_len
            ).X
            source_val_by_source[source] = _source_design(
                loaded.values, target, source, loaded.splits.val_alpha, cfg.lookback, cfg.pred_len
            ).X
            source_cert_by_source[source] = _source_design(
                loaded.values, target, source, loaded.splits.cert, cfg.lookback, cfg.pred_len
            ).X

        for _, screen_row in target_screen.iterrows():
            tasks.append(
                {
                    "target": int(target),
                    "source": int(screen_row["source"]),
                    "s_screen": float(screen_row["s_screen"]),
                    "screen_rank": int(screen_row["screen_rank"]),
                    "group_sources": group_sources,
                    "group_id": f"target_{target}_top{len(group_sources)}",
                    "values": loaded.values,
                    "splits": loaded.splits,
                    "cfg": cfg,
                    "probe_cfg": probe_cfg,
                    "n_jobs": n_jobs,
                    "own_designs": own_by_target[target],
                    "source_train_by_source": source_train_by_source,
                    "source_val_by_source": source_val_by_source,
                    "source_cert_by_source": source_cert_by_source,
                }
            )

    if n_jobs == 1 or len(tasks) <= 1:
        edge_results = [_run_candidate_edge_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks))) as executor:
            edge_results = list(executor.map(_run_candidate_edge_task, tasks))
    pairwise_rows = [pairwise_row for pairwise_row, _ in edge_results]
    candidate_rows = [candidate_row for _, candidate_row in edge_results]

    pairwise_df = pd.DataFrame(pairwise_rows)
    candidate_df = pd.DataFrame(candidate_rows)
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

    run_dir = ensure_run_dir(Path("runs") / "phase5_screen_certify", make_run_id(cfg))
    screen_path = run_dir / "screen_edges.parquet"
    candidate_path = run_dir / "candidate_group_edges.parquet"
    summary_path = run_dir / "summary.json"
    screen.edges.to_parquet(screen_path, index=False)
    candidate_df.to_parquet(candidate_path, index=False)

    summary = {
        "dataset": cfg.dataset,
        "pred_len": int(cfg.pred_len),
        "tier": args.tier,
        "n_targets_screened": int(screen.summary["n_targets_screened"]),
        "top_m": int(top_m),
        "n_screen_edges": int(screen.summary["n_screen_edges"]),
        "spearman_screen_vs_pairwise_delta": spearman_r,
        "spearman_pvalue": spearman_p,
        "n_candidate_group_edges": int(len(candidate_df)),
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
