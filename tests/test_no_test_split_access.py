from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.config import PSVCAConfig, load_config
from psvca.data import scaler as scaler_module
from psvca.data import loader as loader_module
from psvca.data.loader import load_series
from psvca.data.registry import dataset_path, get_dataset_info
from psvca.data.splits import carve_pretest_splits, compute_itransformer_borders


def _assert_phase0_splits(loaded) -> None:
    s = loaded.splits
    assert s.train_fit.start >= s.pre_test.start
    assert s.train_fit.end <= s.pre_test.end
    assert s.val_alpha.start >= s.pre_test.start
    assert s.val_alpha.end <= s.pre_test.end
    assert s.cert.start >= s.pre_test.start
    assert s.cert.end <= s.pre_test.end
    assert s.train_fit.end == s.val_alpha.start
    assert s.val_alpha.end == s.cert.start
    assert not s.cert.overlaps(s.original_test)
    for block in s.stability_blocks:
        assert block.start >= s.pre_test.start
        assert block.end <= s.pre_test.end
        assert not block.overlaps(s.original_test)


def test_smoke_loader_does_not_return_test_rows() -> None:
    loaded = load_series(load_config("smoke"))
    _assert_phase0_splits(loaded)
    assert loaded.values.shape == (loaded.splits.pre_test.length, 4)
    assert loaded.values.shape[0] <= loaded.splits.pre_test.end


def test_data_root_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PSVCA_DATA_ROOT", str(tmp_path))
    cfg = load_config("weather_pl96")
    assert cfg.data_root == str(tmp_path)


def test_scaler_fit_uses_train_fit_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[np.ndarray] = []
    original_fit = scaler_module.StandardScaler.fit

    def spy_fit(self, x):
        seen.append(np.array(x, copy=True))
        return original_fit(self, x)

    monkeypatch.setattr(loader_module.StandardScaler, "fit", spy_fit)
    loaded = load_series(load_config("smoke"))
    assert len(seen) == 1
    fit_data = seen[0]
    assert fit_data.shape[0] == loaded.splits.train_fit.length

    raw_loaded = load_series(load_config("smoke"), include_test=True)
    raw_train_len = raw_loaded.splits.train_fit.length
    assert fit_data.shape[0] == raw_train_len
    np.testing.assert_allclose(loaded.scaler_mean, fit_data.mean(axis=0))
    np.testing.assert_allclose(loaded.scaler_scale, np.where(fit_data.std(axis=0) == 0, 1, fit_data.std(axis=0)))


@pytest.mark.parametrize("config_name", ["smoke", "ettm1_pl96", "etth1_pl96", "weather_pl96"])
def test_all_configs_construct_splits(config_name: str) -> None:
    cfg = load_config(config_name)
    info = get_dataset_info(cfg.dataset)
    if info.relative_path is not None:
        path = dataset_path(cfg.data_root, info)
        if not path.exists():
            pytest.skip(f"real dataset not available: {path}")
    loaded = load_series(cfg)
    _assert_phase0_splits(loaded)
    assert loaded.values.shape[0] == loaded.splits.pre_test.length
    assert len(loaded.channels) == loaded.values.shape[1]


@pytest.mark.parametrize(
    ("dataset_type", "n_rows", "seq_len", "expected"),
    [
        ("ETT_hour", 17420, 96, ((0, 8640), (8544, 11520), (11424, 14400))),
        ("ETT_minute", 69680, 96, ((0, 34560), (34464, 46080), (45984, 57600))),
        ("Custom", 52696, 96, ((0, 36887), (36791, 42157), (42061, 52696))),
    ],
)
def test_itransformer_border_arithmetic(dataset_type, n_rows, seq_len, expected) -> None:
    ranges = compute_itransformer_borders(dataset_type, n_rows, seq_len, pred_len=96)
    assert tuple((r.start, r.end) for r in ranges) == expected


def test_carve_rejects_bad_ratios() -> None:
    original_train, original_val, original_test = compute_itransformer_borders(
        "synthetic", n_rows=720, seq_len=24, pred_len=96
    )
    with pytest.raises(ValueError):
        carve_pretest_splits(
            original_train,
            original_val,
            original_test,
            ratios=(0.5, 0.2, 0.2),
            k_blocks=3,
        )


def test_loader_default_reads_only_pretest_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = 720
    data_dir = tmp_path / "weather"
    data_dir.mkdir()
    csv_path = data_dir / "weather.csv"
    header = ["date"] + [f"c{i}" for i in range(20)] + ["OT"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for i in range(rows):
            values = [f"2020-01-01 {i % 24:02d}:00:00"] + [str(i + j) for j in range(21)]
            f.write(",".join(values) + "\n")

    cfg = PSVCAConfig(
        data_root=str(tmp_path),
        dataset="Weather",
        pred_len=96,
        lookback=24,
        seed=0,
        tier="smoke",
        split_ratios=(0.6, 0.2, 0.2),
        stability_blocks=3,
        alpha_grid=(0.1, 1.0),
        null_method="phase",
        B=20,
    )
    calls = []
    import pandas as pd

    original_read_csv = pd.read_csv

    def spy_read_csv(*args, **kwargs):
        calls.append(kwargs.get("nrows"))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy_read_csv)
    loaded = load_series(cfg)
    assert calls == [loaded.splits.pre_test.end]
    assert loaded.values.shape[0] == loaded.splits.pre_test.length
