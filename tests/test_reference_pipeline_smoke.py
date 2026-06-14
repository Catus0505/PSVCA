from __future__ import annotations

from types import SimpleNamespace

from psvca.config import PSVCAConfig
from psvca.pipeline import reference as reference_pipeline
from scripts.run_synthetic_check import full_splits, make_planted_values


def test_reference_pipeline_synthetic_planted_smoke(monkeypatch, tmp_path) -> None:
    cfg = PSVCAConfig(
        data_root="synthetic://phase6",
        dataset="synthetic_planted",
        pred_len=1,
        lookback=6,
        seed=2026,
        tier="sanity",
        split_ratios=(0.5, 0.25, 0.25),
        stability_blocks=1,
        alpha_grid=(0.01, 0.1, 1.0, 10.0),
        null_method="phase",
        B=39,
    )

    def fake_load_series(_cfg):
        return SimpleNamespace(
            values=make_planted_values(_cfg.seed),
            channels=("target", "strong_source", "shared_period_decoy", "zero_source"),
            splits=full_splits(),
        )

    monkeypatch.setattr(reference_pipeline, "load_series", fake_load_series)
    edges, summary, run_dir = reference_pipeline.run_reference_pipeline(
        cfg,
        tier="sanity",
        n_jobs=1,
        output_root=tmp_path,
    )

    assert run_dir.exists()
    assert {"e_certified", "fdr_pass", "stability_pass"}.issubset(edges.columns)
    strong = edges[(edges["target"] == 0) & (edges["source"] == 1)].iloc[0]
    zero = edges[(edges["target"] == 0) & (edges["source"] == 3)].iloc[0]
    assert bool(strong["e_certified"])
    assert not bool(zero["e_certified"])
    assert summary["n_edges"] == len(edges)
