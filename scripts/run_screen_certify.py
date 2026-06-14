from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.certify.probe import PairwiseProbeConfig, probe_candidate_group, probe_pairwise
from psvca.config import load_config
from psvca.data.loader import load_series
from psvca.io.artifacts import ensure_run_dir, make_run_id
from psvca.linalg.design import make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate
from psvca.screen.value_screen import ValueScreenConfig, run_value_screen


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

    pairwise_rows = []
    candidate_rows = []
    passed = screen.edges[screen.edges["passed_screen"]].copy()
    for target in targets:
        own_train, own_val, own_cert = own_by_target[target]
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
            source = int(screen_row["source"])
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
                surrogate_bank=_surrogate_bank(
                    loaded.values,
                    target=target,
                    source=source,
                    splits=loaded.splits,
                    lookback=cfg.lookback,
                    horizon=cfg.pred_len,
                    B=B,
                    seed=cfg.seed,
                    dataset=cfg.dataset,
                ),
                config=probe_cfg,
            )
            pairwise_rows.append(_pairwise_row(pairwise))
            candidate = probe_candidate_group(
                target=target,
                source=source,
                group_sources=group_sources,
                y_train=own_train.y,
                y_val=own_val.y,
                y_cert=own_cert.y,
                own_train=own_train.X,
                own_val=own_val.X,
                own_cert=own_cert.X,
                source_train_by_source=source_train_by_source,
                source_val_by_source=source_val_by_source,
                source_cert_by_source=source_cert_by_source,
                surrogate_bank=_surrogate_bank(
                    loaded.values,
                    target=target,
                    source=source,
                    splits=loaded.splits,
                    lookback=cfg.lookback,
                    horizon=cfg.pred_len,
                    B=B,
                    seed=cfg.seed,
                    dataset=cfg.dataset,
                ),
                group_id=f"target_{target}_top{len(group_sources)}",
                config=probe_cfg,
            )
            candidate_rows.append(_candidate_group_row(candidate, screen_row))

    pairwise_df = pd.DataFrame(pairwise_rows)
    candidate_df = pd.DataFrame(candidate_rows)
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
