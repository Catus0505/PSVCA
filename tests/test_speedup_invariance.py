from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.admission.aggregate import aggregate_certified_edges
from psvca.certify.fdr import FDRConfig, apply_bh_fdr
from psvca.certify.probe import PairwiseProbeConfig, fit_baseline_cache, probe_pairwise
from psvca.certify.stability import StabilityConfig, apply_stability
from psvca.linalg.design import make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate
from scripts.run_synthetic_check import EDGE_TYPES, full_splits, make_planted_values


def _own_design(values, target: int, split, lookback: int, horizon: int):
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


def _source_design(values, target: int, source: int, split, lookback: int, horizon: int):
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


def _phase_surrogates_by_source(values, *, sources: tuple[int, ...], B: int, seed: int):
    return {
        int(source): [
            make_phase_surrogate(
                values[:, int(source)],
                source_idx=int(source),
                surrogate_id=surrogate_id,
                seed=seed,
                dataset="speedup_invariance",
                split="pre_test",
            ).values
            for surrogate_id in range(B)
        ]
        for source in sources
    }


def _surrogate_bank(
    values,
    *,
    target: int,
    source: int,
    splits,
    lookback: int,
    horizon: int,
    B: int,
    seed: int,
    source_surrogates=None,
):
    bank = []
    for surrogate_id in range(B):
        if source_surrogates is None:
            surrogate = make_phase_surrogate(
                values[:, source],
                source_idx=source,
                surrogate_id=surrogate_id,
                seed=seed,
                dataset="speedup_invariance",
                split="pre_test",
            ).values
        else:
            surrogate = source_surrogates[int(source)][surrogate_id]
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


def _probe_edges(
    *,
    skip_null_on_fail: bool,
    use_own_cache: bool = False,
    share_surrogates: bool = False,
    targets: tuple[int, ...] = (0,),
) -> pd.DataFrame:
    values = make_planted_values(seed=2026)
    splits = full_splits()
    lookback = 6
    horizon = 1
    B = 12
    seed = 2026
    cfg = PairwiseProbeConfig(
        alphas=(0.01, 0.1, 1.0, 10.0),
        B=B,
        seed=seed,
        null_method="phase",
        alpha_rule="val_grid",
        skip_null_on_fail=skip_null_on_fail,
    )
    all_sources = tuple(
        sorted({source for target in targets for source in range(values.shape[1]) if source != target})
    )
    source_surrogates = (
        _phase_surrogates_by_source(values, sources=all_sources, B=B, seed=seed)
        if share_surrogates
        else None
    )

    rows = []
    for target in targets:
        own_train = _own_design(values, target, splits.train_fit, lookback, horizon)
        own_val = _own_design(values, target, splits.val_alpha, lookback, horizon)
        own_cert = _own_design(values, target, splits.cert, lookback, horizon)
        own_cache = None
        if use_own_cache:
            own_cache = fit_baseline_cache(
                sources=(),
                X_train=own_train.X,
                y_train=own_train.y,
                X_val=own_val.X,
                y_val=own_val.y,
                X_cert=own_cert.X,
                y_cert=own_cert.y,
                alphas=cfg.alphas,
                alpha_rule=cfg.alpha_rule,
                variance_eps=cfg.variance_eps,
            )
        for source in range(values.shape[1]):
            if source == target:
                continue
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
                own_cache=own_cache,
                surrogate_bank=_surrogate_bank(
                    values,
                    target=target,
                    source=source,
                    splits=splits,
                    lookback=lookback,
                    horizon=horizon,
                    B=B,
                    seed=seed,
                    source_surrogates=source_surrogates,
                ),
                config=cfg,
            )
            row = asdict(result)
            row.pop("delta_null")
            row.pop("alpha_null")
            row["edge_type"] = EDGE_TYPES.get((target, source), "unknown")
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "source"]).reset_index(drop=True)


def _certified_set(edges: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(row.target), int(row.source))
        for row in edges.itertuples(index=False)
        if bool(row.e_certified)
    }


def _aggregate(edges: pd.DataFrame) -> pd.DataFrame:
    fdr = apply_bh_fdr(edges, FDRConfig(q=0.1, min_B_for_formal=200)).edges
    stable = apply_stability(fdr, [fdr], StabilityConfig(min_fraction=1.0)).edges
    return aggregate_certified_edges(stable).edges.sort_values(["target", "source"]).reset_index(drop=True)


def test_skip_null_on_first_gate_failure_preserves_certified_set() -> None:
    full = _aggregate(_probe_edges(skip_null_on_fail=False))
    skipped = _aggregate(_probe_edges(skip_null_on_fail=True))

    assert _certified_set(skipped) == _certified_set(full)
    assert int(skipped["skipped_null"].sum()) > 0
    skipped_rows = skipped[skipped["skipped_null"]]
    assert skipped_rows["p_value"].eq(1.0).all()
    assert not skipped_rows["certified_candidate"].any()


def test_own_fit_cache_preserves_certified_set_and_reduces_own_svd(monkeypatch) -> None:
    from psvca.certify import probe as probe_module

    original_fit = probe_module.fit_ridge_path
    own_fit_count = 0

    def counting_fit(X, y, alphas):
        nonlocal own_fit_count
        if X.shape[1] == 6:
            own_fit_count += 1
        return original_fit(X, y, alphas)

    monkeypatch.setattr(probe_module, "fit_ridge_path", counting_fit)
    uncached = _aggregate(_probe_edges(skip_null_on_fail=False, use_own_cache=False))
    uncached_own_fits = own_fit_count
    own_fit_count = 0
    cached = _aggregate(_probe_edges(skip_null_on_fail=False, use_own_cache=True))
    cached_own_fits = own_fit_count

    assert _certified_set(cached) == _certified_set(uncached)
    assert uncached_own_fits == 3
    assert cached_own_fits == 1


def test_shared_phase_surrogates_preserve_p_values_bitwise() -> None:
    per_edge = _probe_edges(skip_null_on_fail=False, share_surrogates=False)
    shared = _probe_edges(skip_null_on_fail=False, share_surrogates=True)

    assert _certified_set(_aggregate(shared)) == _certified_set(_aggregate(per_edge))
    np.testing.assert_array_equal(
        shared["p_value"].to_numpy(dtype=float),
        per_edge["p_value"].to_numpy(dtype=float),
    )


def test_shared_phase_surrogates_generate_once_per_source(monkeypatch) -> None:
    module = sys.modules[__name__]
    original = module.make_phase_surrogate
    call_count = 0

    def counting_surrogate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "make_phase_surrogate", counting_surrogate)
    _probe_edges(skip_null_on_fail=False, share_surrogates=False, targets=(0, 1))
    per_edge_count = call_count
    call_count = 0
    _probe_edges(skip_null_on_fail=False, share_surrogates=True, targets=(0, 1))
    shared_count = call_count

    assert per_edge_count == 72
    assert shared_count == 48
