from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.certify.probe import PairwiseProbeConfig, probe_pairwise
from psvca.config import load_config
from psvca.data.loader import load_series
from psvca.io.artifacts import ensure_run_dir, make_run_id
from psvca.linalg.design import make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", required=True)
    args = parser.parse_args()

    if args.tier != "sanity":
        raise SystemExit("Phase 4 reference only supports --tier sanity")
    cfg = load_config(args.config)
    loaded = load_series(cfg)
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
    own_by_target = {}
    rows = []
    for target in range(n_channels):
        own_by_target[target] = (
            _own_design(loaded.values, target, loaded.splits.train_fit, cfg.lookback, cfg.pred_len),
            _own_design(loaded.values, target, loaded.splits.val_alpha, cfg.lookback, cfg.pred_len),
            _own_design(loaded.values, target, loaded.splits.cert, cfg.lookback, cfg.pred_len),
        )

    for target in range(n_channels):
        own_train, own_val, own_cert = own_by_target[target]
        for source in range(n_channels):
            if source == target:
                continue
            source_train = _source_design(
                loaded.values, target, source, loaded.splits.train_fit, cfg.lookback, cfg.pred_len
            )
            source_val = _source_design(
                loaded.values, target, source, loaded.splits.val_alpha, cfg.lookback, cfg.pred_len
            )
            source_cert = _source_design(
                loaded.values, target, source, loaded.splits.cert, cfg.lookback, cfg.pred_len
            )
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
                    loaded.values,
                    target=target,
                    source=source,
                    splits=loaded.splits,
                    lookback=cfg.lookback,
                    horizon=cfg.pred_len,
                    B=cfg.B,
                    seed=cfg.seed,
                    dataset=cfg.dataset,
                ),
                config=probe_cfg,
            )
            rows.append(_row(result))
            print(
                f"target={target} source={source} delta={result.delta_true:.6g} "
                f"aligned={result.aligned_gain:.6g} p={result.p_value:.6g} "
                f"candidate={result.certified_candidate}"
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
    print(f"  output={out_path}")


if __name__ == "__main__":
    main()
