from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    relative_path: str | None
    dataset_type: str
    expected_channels: int | None
    target_policy: str
    default_pred_lens: tuple[int, ...]
    target: str = "OT"
    features: str = "M"


DATASETS: dict[str, DatasetInfo] = {
    "ETTm1": DatasetInfo(
        name="ETTm1",
        relative_path="ETT-small/ETTm1.csv",
        dataset_type="ETT_minute",
        expected_channels=7,
        target_policy="all_except_date",
        default_pred_lens=(96, 192, 336, 720),
    ),
    "ETTh1": DatasetInfo(
        name="ETTh1",
        relative_path="ETT-small/ETTh1.csv",
        dataset_type="ETT_hour",
        expected_channels=7,
        target_policy="all_except_date",
        default_pred_lens=(96, 192, 336, 720),
    ),
    "Weather": DatasetInfo(
        name="Weather",
        relative_path="weather/weather.csv",
        dataset_type="Custom",
        expected_channels=21,
        target_policy="itransformer_custom_target_last",
        default_pred_lens=(96, 192, 336, 720),
    ),
    "smoke": DatasetInfo(
        name="smoke",
        relative_path=None,
        dataset_type="synthetic",
        expected_channels=4,
        target_policy="synthetic_all",
        default_pred_lens=(96,),
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    try:
        return DATASETS[name]
    except KeyError as exc:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {name!r}; known datasets: {known}") from exc


def dataset_path(data_root: str | Path, info: DatasetInfo) -> Path:
    if info.relative_path is None:
        raise ValueError(f"dataset {info.name} has no CSV path")
    rel = Path(info.relative_path)
    if rel.is_absolute():
        raise ValueError(f"dataset path for {info.name} must be relative: {rel}")
    return Path(data_root) / rel
