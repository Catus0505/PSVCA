from __future__ import annotations

import argparse
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

from psvca.certify.probe import PairwiseProbeConfig, normalize_n_jobs, probe_pairwise
from psvca.config import load_config
from psvca.data.loader import load_series
from psvca.io.artifacts import ensure_run_dir, make_run_id
from psvca.linalg.design import make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate


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


def _row(result) -> dict:
    data = asdict(result)
    data.pop("delta_null")
    data.pop("alpha_null")
    data["mode"] = "pairwise"
    data["alpha_rule"] = "val_grid"
    return data


def _run_pairwise_edge_task(task: dict) -> dict:
    values = task["values"]
    cfg = task["cfg"]
    splits = task["splits"]
    target = int(task["target"])
    source = int(task["source"])
    own_train = _own_design(values, target, splits.train_fit, cfg.lookback, cfg.pred_len)
    own_val = _own_design(values, target, splits.val_alpha, cfg.lookback, cfg.pred_len)
    own_cert = _own_design(values, target, splits.cert, cfg.lookback, cfg.pred_len)
    source_train = _source_design(values, target, source, splits.train_fit, cfg.lookback, cfg.pred_len)
    source_val = _source_design(values, target, source, splits.val_alpha, cfg.lookback, cfg.pred_len)
    source_cert = _source_design(values, target, source, splits.cert, cfg.lookback, cfg.pred_len)
    result = probe_pairwise(
        target=target,
        source=source,
        y_train=own_train.y,
        y_val=own_val.y,
        y_cert=own_cert.y,
        own_train=own_train.X,
        own_val=own_val.X,
        own_cert=own_cert.X,
        source_train=source_train.X,
        source_val=source_val.X,
        source_cert=source_cert.X,
        surrogate_bank=_surrogate_bank(
            values,
            target=target,
            source=source,
            splits=splits,
            lookback=cfg.lookback,
            horizon=cfg.pred_len,
            B=cfg.B,
            seed=cfg.seed,
            dataset=cfg.dataset,
        ),
        config=task["probe_cfg"],
    )
    return _row(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    if args.tier != "sanity":
        raise SystemExit("Phase 4 reference only supports --tier sanity")
    cfg = load_config(args.config)
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
