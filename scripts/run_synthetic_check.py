from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
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

from psvca.admission.aggregate import aggregate_certified_edges
from psvca.certify.fdr import FDRConfig, apply_bh_fdr
from psvca.certify.probe import PairwiseProbeConfig, normalize_n_jobs, probe_pairwise
from psvca.certify.stability import StabilityConfig, apply_stability
from psvca.config import load_config
from psvca.io.artifacts import ensure_run_dir, make_run_id
from psvca.linalg.design import make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate


@dataclass(frozen=True)
class Range:
    start: int
    end: int


@dataclass(frozen=True)
class ProbeSplits:
    train_fit: Range
    val_alpha: Range
    cert: Range


EDGE_TYPES = {
    (0, 1): "strong",
    (0, 2): "shared_period_decoy",
    (0, 3): "zero",
}


def make_planted_values(seed: int, n: int = 380) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    period = np.sin(2.0 * np.pi * t / 31.0)
    strong = np.zeros(n, dtype=np.float64)
    zero = np.zeros(n, dtype=np.float64)
    target = np.zeros(n, dtype=np.float64)
    strong_noise = rng.normal(scale=0.7, size=n)
    zero_noise = rng.normal(scale=0.8, size=n)
    target_noise = rng.normal(scale=0.18, size=n)
    decoy = period + rng.normal(scale=0.08, size=n)
    for idx in range(1, n):
        strong[idx] = 0.55 * strong[idx - 1] + strong_noise[idx]
        zero[idx] = 0.45 * zero[idx - 1] + zero_noise[idx]
        target[idx] = (
            0.45 * target[idx - 1]
            + 1.35 * strong[idx - 1]
            + 0.75 * period[idx - 1]
            + target_noise[idx]
        )
    values = np.column_stack([target, strong, decoy, zero])
    return (values - values.mean(axis=0)) / values.std(axis=0)


def full_splits() -> ProbeSplits:
    return ProbeSplits(train_fit=Range(40, 170), val_alpha=Range(170, 230), cert=Range(230, 350))


def block_splits() -> list[ProbeSplits]:
    return [
        ProbeSplits(train_fit=Range(40, 150), val_alpha=Range(150, 205), cert=Range(205, 250)),
        ProbeSplits(train_fit=Range(40, 180), val_alpha=Range(180, 235), cert=Range(235, 290)),
        ProbeSplits(train_fit=Range(40, 210), val_alpha=Range(210, 265), cert=Range(265, 340)),
    ]


def _own_design(values: np.ndarray, target: int, split: Range, lookback: int, horizon: int):
    return make_lagged_design(
        values,
        target=target,
        sources=(),
        lookback=lookback,
        horizon=horizon,
        y_start=split.start,
        y_end=split.end,
        include_own=True,
    )


def _source_design(
    values: np.ndarray,
    target: int,
    source: int,
    split: Range,
    lookback: int,
    horizon: int,
):
    return make_lagged_design(
        values,
        target=target,
        sources=(source,),
        lookback=lookback,
        horizon=horizon,
        y_start=split.start,
        y_end=split.end,
        include_own=False,
    )


def _surrogate_bank(
    values: np.ndarray,
    *,
    target: int,
    source: int,
    splits: ProbeSplits,
    lookback: int,
    horizon: int,
    B: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    bank = []
    for surrogate_id in range(B):
        surrogate = make_phase_surrogate(
            values[:, source],
            source_idx=source,
            surrogate_id=surrogate_id,
            seed=seed,
            dataset="synthetic_planted",
            split="pre_test",
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
    data["edge_type"] = EDGE_TYPES.get((data["target"], data["source"]), "unknown")
    return data


def probe_synthetic_edges(
    values: np.ndarray,
    *,
    splits: ProbeSplits,
    lookback: int,
    horizon: int,
    alphas: tuple[float, ...],
    B: int,
    seed: int,
) -> pd.DataFrame:
    target = 0
    probe_cfg = PairwiseProbeConfig(
        alphas=alphas,
        B=B,
        seed=seed,
        null_method="phase",
        alpha_rule="val_grid",
    )
    own_train = _own_design(values, target, splits.train_fit, lookback, horizon)
    own_val = _own_design(values, target, splits.val_alpha, lookback, horizon)
    own_cert = _own_design(values, target, splits.cert, lookback, horizon)
    rows = []
    for source in (1, 2, 3):
        source_train = _source_design(values, target, source, splits.train_fit, lookback, horizon)
        source_val = _source_design(values, target, source, splits.val_alpha, lookback, horizon)
        source_cert = _source_design(values, target, source, splits.cert, lookback, horizon)
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
                lookback=lookback,
                horizon=horizon,
                B=B,
                seed=seed,
            ),
            config=probe_cfg,
        )
        rows.append(_row(result))
    return pd.DataFrame(rows).sort_values(["target", "source"]).reset_index(drop=True)


def run_phase6_synthetic(
    *,
    seed: int,
    lookback: int,
    horizon: int,
    alphas: tuple[float, ...],
    B: int,
    fdr_q: float = 0.1,
    stability_min_fraction: float = 0.67,
) -> tuple[pd.DataFrame, dict]:
    values = make_planted_values(seed)
    main_edges = probe_synthetic_edges(
        values,
        splits=full_splits(),
        lookback=lookback,
        horizon=horizon,
        alphas=alphas,
        B=B,
        seed=seed,
    )
    fdr_result = apply_bh_fdr(main_edges, FDRConfig(q=fdr_q, min_B_for_formal=200))
    block_results = []
    for splits in block_splits():
        block = probe_synthetic_edges(
            values,
            splits=splits,
            lookback=lookback,
            horizon=horizon,
            alphas=alphas,
            B=B,
            seed=seed,
        )
        block_results.append(apply_bh_fdr(block, FDRConfig(q=fdr_q, min_B_for_formal=200)).edges)
    stability_result = apply_stability(
        fdr_result.edges,
        block_results,
        StabilityConfig(min_fraction=stability_min_fraction),
    )
    aggregate = aggregate_certified_edges(stability_result.edges)
    edges = aggregate.edges.sort_values(["target", "source"]).reset_index(drop=True)
    strong = edges["edge_type"] == "strong"
    zero = edges["edge_type"] == "zero"
    decoy = edges["edge_type"] == "shared_period_decoy"
    summary = {
        **aggregate.summary,
        "strong_edges_total": int(strong.sum()),
        "strong_edges_recovered": int(edges.loc[strong, "e_certified"].sum()),
        "zero_edges_fdr_pass": int(edges.loc[zero, "fdr_pass"].sum()),
        "decoy_edges_e_certified": int(edges.loc[decoy, "e_certified"].sum()),
        "fdr_q": float(fdr_q),
        "stability_min_fraction": float(stability_min_fraction),
        "B": int(B),
        "fdr_underpowered": bool(edges["fdr_underpowered"].any()),
    }
    return edges, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="synthetic_planted")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    normalize_n_jobs(args.n_jobs)
    cfg = load_config(args.config)
    edges, summary = run_phase6_synthetic(
        seed=cfg.seed,
        lookback=cfg.lookback,
        horizon=cfg.pred_len,
        alphas=cfg.alpha_grid,
        B=cfg.B,
    )
    summary["n_jobs_requested"] = int(args.n_jobs)
    summary["n_jobs_effective"] = int(normalize_n_jobs(args.n_jobs))
    run_dir = ensure_run_dir(Path("runs") / "phase6_synthetic_check", make_run_id(cfg))
    edges_path = run_dir / "edges_phase6.parquet"
    summary_path = run_dir / "summary.json"
    edges.to_parquet(edges_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("summary:")
    for key, value in summary.items():
        print(f"  {key}={value}")
    print(f"  edges_phase6={edges_path}")
    print(f"  summary={summary_path}")
    if summary["strong_edges_recovered"] != summary["strong_edges_total"]:
        raise SystemExit("synthetic check failed: strong edge was not recovered")
    if summary["zero_edges_fdr_pass"] != 0:
        raise SystemExit("synthetic check failed: zero edge passed FDR")
    if summary["decoy_edges_e_certified"] != 0:
        raise SystemExit("synthetic check failed: decoy edge was certified")


if __name__ == "__main__":
    main()
