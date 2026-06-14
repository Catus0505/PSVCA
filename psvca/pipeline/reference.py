from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from psvca.admission.aggregate import aggregate_certified_edges
from psvca.certify.fdr import FDRConfig, apply_bh_fdr
from psvca.certify.probe import PairwiseProbeConfig, normalize_n_jobs, probe_pairwise
from psvca.certify.stability import StabilityConfig, apply_stability
from psvca.config import PSVCAConfig, config_hash
from psvca.data.loader import load_series
from psvca.io.artifacts import ensure_run_dir, get_git_hash, make_run_id
from psvca.io.schema import SCHEMA_VERSION
from psvca.linalg.design import make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate


# Weather N=21 is allowed for exact all-pair reference. ECL N=321 and
# Traffic N=862 exceed this cap and must use screen_certify (dead rule 2).
MAX_EXACT_N = 32
MIN_FORMAL_B = 200


def _bounds(split) -> tuple[int, int]:
    return int(split.start), int(split.end)


def _own_design(values: np.ndarray, target: int, split, lookback: int, horizon: int):
    start, end = _bounds(split)
    return make_lagged_design(values, target, (), lookback, horizon, start, end, include_own=True)


def _source_design(values: np.ndarray, target: int, source: int, split, lookback: int, horizon: int):
    start, end = _bounds(split)
    return make_lagged_design(values, target, (source,), lookback, horizon, start, end, include_own=False)


def _surrogate_bank(values, *, target, source, splits, lookback, horizon, B, seed, dataset):
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


def _row(result, *, cfg: PSVCAConfig | None = None, run_id: str | None = None, git_hash: str | None = None) -> dict:
    from dataclasses import asdict

    data = asdict(result)
    data.pop("delta_null")
    data.pop("alpha_null")
    data["mode"] = "pairwise"
    data["alpha_rule"] = "val_grid"
    if cfg is not None and run_id is not None and git_hash is not None:
        data.update(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "config_hash": config_hash(cfg),
                "git_hash": git_hash,
                "seed": int(cfg.seed),
                "dataset": cfg.dataset,
                "pred_len": int(cfg.pred_len),
                "tier": cfg.tier,
            }
        )
    return data


def pairwise_edge_task(task: dict) -> dict:
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
    if "run_id" in task and "git_hash" in task:
        return _row(result, cfg=cfg, run_id=task["run_id"], git_hash=task["git_hash"])
    return _row(result)


def _edge_task(task: dict) -> dict:
    return pairwise_edge_task(task)


def _cert_blocks(splits, k: int):
    start, end = int(splits.cert.start), int(splits.cert.end)
    edges = [start + (end - start) * i // k for i in range(k + 1)]
    blocks = []
    for i in range(k):
        block = replace(splits, cert=replace(splits.cert, start=edges[i], end=edges[i + 1]))
        if block.cert.end > block.cert.start:
            blocks.append(block)
    return blocks


def run_reference_pipeline(
    cfg: PSVCAConfig,
    *,
    tier: str | None = None,
    n_jobs: int = 1,
    output_root: str | Path = "runs/phase7_reference",
) -> tuple[pd.DataFrame, dict, Path]:
    effective_cfg = replace(cfg, tier=tier or cfg.tier)
    n_jobs_eff = normalize_n_jobs(n_jobs)
    loaded = load_series(effective_cfg)
    n_channels = loaded.values.shape[1]
    if n_channels > MAX_EXACT_N:
        raise SystemExit(
            f"exact reference 仅限小 N,大 N 走 screen_certify: N={n_channels}, max={MAX_EXACT_N}"
        )
    requested_tier = tier or cfg.tier
    effective_B = max(effective_cfg.B, MIN_FORMAL_B) if requested_tier == "formal" else effective_cfg.B
    probe_cfg = PairwiseProbeConfig(
        alphas=effective_cfg.alpha_grid,
        B=effective_B,
        seed=effective_cfg.seed,
        null_method=effective_cfg.null_method,
        alpha_rule="val_grid",
    )
    run_id = make_run_id(effective_cfg)
    git_hash = get_git_hash()
    tasks = [
        {
            "values": loaded.values,
            "splits": loaded.splits,
            "cfg": effective_cfg,
            "probe_cfg": probe_cfg,
            "target": target,
            "source": source,
            "run_id": run_id,
            "git_hash": git_hash,
        }
        for target in range(n_channels)
        for source in range(n_channels)
        if source != target
    ]
    if n_jobs_eff == 1 or len(tasks) <= 1:
        rows = [_edge_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(n_jobs_eff, len(tasks))) as executor:
            rows = list(executor.map(_edge_task, tasks))
    full_edges = pd.DataFrame(rows).sort_values(["target", "source"]).reset_index(drop=True)
    fdr_edges = apply_bh_fdr(full_edges, FDRConfig(q=0.1, min_B_for_formal=200)).edges
    block_edges = []
    for block_splits in _cert_blocks(loaded.splits, max(1, effective_cfg.stability_blocks)):
        block_tasks = [{**task, "splits": block_splits} for task in tasks]
        block_rows = [_edge_task(task) for task in block_tasks]
        block_edges.append(apply_bh_fdr(pd.DataFrame(block_rows), FDRConfig(q=0.1, min_B_for_formal=200)).edges)
    stable = apply_stability(fdr_edges, block_edges, StabilityConfig()).edges
    aggregate = aggregate_certified_edges(stable).edges
    run_dir = ensure_run_dir(output_root, run_id)
    out_path = run_dir / "edges.parquet"
    aggregate.to_parquet(out_path, index=False)
    summary = {
        "dataset": effective_cfg.dataset,
        "pred_len": int(effective_cfg.pred_len),
        "tier": effective_cfg.tier,
        "n_edges": int(len(aggregate)),
        "n_e_certified": int(aggregate["e_certified"].sum()),
        "effective_B": int(effective_B),
        "n_jobs_requested": int(n_jobs),
        "n_jobs_effective": int(n_jobs_eff),
        "cpu_count": int(os.cpu_count() or 1),
        "output": str(out_path),
    }
    (run_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return aggregate, summary, run_dir
