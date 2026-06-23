from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from psvca.certify.probe import (
    BaselineFitCache,
    PairwiseProbeConfig,
    fit_baseline_cache,
    probe_candidate_group,
    probe_pairwise,
)
from psvca.linalg.design import DesignMatrix, make_lagged_design
from psvca.nulls.phase_surrogate import make_phase_surrogate


@dataclass(frozen=True)
class DesignBlocks:
    train: np.ndarray
    val: np.ndarray
    cert: np.ndarray


class CertificationDriver:
    def __init__(
        self,
        *,
        values: np.ndarray,
        splits,
        lookback: int,
        horizon: int,
        probe_config: PairwiseProbeConfig,
        seed: int,
        dataset: str,
        n_jobs: int = 1,
        backend: str = "cpu",
    ) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.splits = splits
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.probe_config = probe_config
        self.seed = int(seed)
        self.dataset = str(dataset)
        self.n_jobs = int(n_jobs)
        self.backend = str(backend)
        if self.backend not in {"cpu", "gpu"}:
            raise ValueError(f"unsupported backend: {self.backend!r}")
        self.n_channels = int(self.values.shape[1])
        self._own_designs: dict[int, tuple[DesignMatrix, DesignMatrix, DesignMatrix]] = {}
        self._own_caches: dict[int, BaselineFitCache] = {}
        self._baseline_caches: dict[tuple[int, tuple[int, ...]], BaselineFitCache] = {}
        self._source_designs: dict[int, DesignBlocks] = {}
        self._source_surrogates: dict[int, list[np.ndarray]] = {}
        self._surrogate_designs: dict[int, list[DesignBlocks]] = {}

    def own_designs(self, target: int) -> tuple[DesignMatrix, DesignMatrix, DesignMatrix]:
        target = int(target)
        if target not in self._own_designs:
            self._own_designs[target] = (
                self._own_design(target, self.splits.train_fit),
                self._own_design(target, self.splits.val_alpha),
                self._own_design(target, self.splits.cert),
            )
        return self._own_designs[target]

    def own_cache(self, target: int) -> BaselineFitCache:
        target = int(target)
        if target not in self._own_caches:
            self._own_caches[target] = self.baseline_cache(target, ())
        return self._own_caches[target]

    def baseline_cache(self, target: int, sources: tuple[int, ...]) -> BaselineFitCache:
        target = int(target)
        sources = tuple(int(s) for s in sources)
        key = (target, sources)
        if key not in self._baseline_caches:
            own_train, own_val, own_cert = self.own_designs(target)
            train_blocks, val_blocks, cert_blocks = self.source_design_dicts(sources)
            X_train = _stack_design(own_train.X, train_blocks, sources)
            X_val = _stack_design(own_val.X, val_blocks, sources)
            X_cert = _stack_design(own_cert.X, cert_blocks, sources)
            self._baseline_caches[key] = fit_baseline_cache(
                sources=sources,
                X_train=X_train,
                y_train=own_train.y,
                X_val=X_val,
                y_val=own_val.y,
                X_cert=X_cert,
                y_cert=own_cert.y,
                alphas=self.probe_config.alphas,
                alpha_rule=self.probe_config.alpha_rule,
                variance_eps=self.probe_config.variance_eps,
            )
        return self._baseline_caches[key]

    def source_designs(self, source: int) -> DesignBlocks:
        source = int(source)
        if source not in self._source_designs:
            target = self._dummy_target(source)
            self._source_designs[source] = DesignBlocks(
                train=self._source_design(target, source, self.splits.train_fit).X,
                val=self._source_design(target, source, self.splits.val_alpha).X,
                cert=self._source_design(target, source, self.splits.cert).X,
            )
        return self._source_designs[source]

    def source_design_dicts(
        self, group_sources: tuple[int, ...]
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
        train: dict[int, np.ndarray] = {}
        val: dict[int, np.ndarray] = {}
        cert: dict[int, np.ndarray] = {}
        for source in group_sources:
            blocks = self.source_designs(source)
            train[int(source)] = blocks.train
            val[int(source)] = blocks.val
            cert[int(source)] = blocks.cert
        return train, val, cert

    def phase_surrogates(self, source: int) -> list[np.ndarray]:
        source = int(source)
        if source not in self._source_surrogates:
            self._source_surrogates[source] = [
                make_phase_surrogate(
                    self.values[:, source],
                    source_idx=source,
                    surrogate_id=surrogate_id,
                    seed=self.seed,
                    dataset=self.dataset,
                    split="pre_test",
                    cache_dir=None,
                ).values
                for surrogate_id in range(self.probe_config.B)
            ]
        return self._source_surrogates[source]

    def surrogate_designs(self, source: int) -> list[DesignBlocks]:
        source = int(source)
        if source not in self._surrogate_designs:
            target = self._dummy_target(source)
            designs = []
            for surrogate in self.phase_surrogates(source):
                s_values = self.values.copy()
                s_values[:, source] = surrogate
                designs.append(
                    DesignBlocks(
                        train=self._source_design(
                            target, source, self.splits.train_fit, values=s_values
                        ).X,
                        val=self._source_design(
                            target, source, self.splits.val_alpha, values=s_values
                        ).X,
                        cert=self._source_design(target, source, self.splits.cert, values=s_values).X,
                    )
                )
            self._surrogate_designs[source] = designs
        return self._surrogate_designs[source]

    def surrogate_bank(self, *, target: int, source: int):
        source = int(source)
        del target
        for blocks in self.surrogate_designs(source):
            yield (blocks.train, blocks.val, blocks.cert)

    def probe_pairwise_row(self, *, target: int, source: int, metadata: dict | None = None) -> dict:
        target = int(target)
        source = int(source)
        own_train, own_val, own_cert = self.own_designs(target)
        source_blocks = self.source_designs(source)
        result = probe_pairwise(
            target=target,
            source=source,
            y_train=own_train.y,
            y_val=own_val.y,
            y_cert=own_cert.y,
            own_train=own_train.X,
            own_val=own_val.X,
            own_cert=own_cert.X,
            source_train=source_blocks.train,
            source_val=source_blocks.val,
            source_cert=source_blocks.cert,
            own_cache=self.own_cache(target),
            surrogate_bank=self.surrogate_bank(target=target, source=source),
            config=self.probe_config,
        )
        row = _result_row(result)
        row["mode"] = "pairwise"
        if metadata:
            row.update(metadata)
        return row

    def probe_candidate_group_row(
        self,
        *,
        target: int,
        source: int,
        group_sources: tuple[int, ...],
        group_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        target = int(target)
        source = int(source)
        group_sources = tuple(int(s) for s in group_sources)
        own_train, own_val, own_cert = self.own_designs(target)
        source_train, source_val, source_cert = self.source_design_dicts(group_sources)
        other_sources = tuple(s for s in group_sources if s != source)
        result = probe_candidate_group(
            target=target,
            source=source,
            group_sources=group_sources,
            y_train=own_train.y,
            y_val=own_val.y,
            y_cert=own_cert.y,
            own_train=own_train.X,
            own_val=own_val.X,
            own_cert=own_cert.X,
            source_train_by_source=source_train,
            source_val_by_source=source_val,
            source_cert_by_source=source_cert,
            reduced_cache=self.baseline_cache(target, other_sources),
            surrogate_bank=self.surrogate_bank(target=target, source=source),
            group_id=group_id,
            config=self.probe_config,
        )
        row = _result_row(result)
        if metadata:
            row.update(metadata)
        return row

    def pairwise_edges(self, targets, *, metadata_for=None) -> pd.DataFrame:
        targets = tuple(int(target) for target in targets)
        if self.n_jobs > 1 and len(targets) > 1:
            tasks = [
                {
                    "values": self.values,
                    "splits": self.splits,
                    "lookback": self.lookback,
                    "horizon": self.horizon,
                    "probe_config": self.probe_config,
                    "seed": self.seed,
                    "dataset": self.dataset,
                    "target": target,
                    "sources": tuple(source for source in range(self.n_channels) if source != target),
                    "metadata_by_source": {
                        source: metadata_for(target, source) if metadata_for else None
                        for source in range(self.n_channels)
                        if source != target
                    },
                }
                for target in targets
            ]
            rows = _parallel_target_rows(_pairwise_target_task, tasks, self.n_jobs)
            return pd.DataFrame(rows).sort_values(["target", "source"]).reset_index(drop=True)

        rows = []
        for target in targets:
            for source in range(self.n_channels):
                if int(source) == int(target):
                    continue
                metadata = metadata_for(int(target), int(source)) if metadata_for else None
                rows.append(
                    self.probe_pairwise_row(
                        target=int(target),
                        source=int(source),
                        metadata=metadata,
                    )
                )
        return pd.DataFrame(rows).sort_values(["target", "source"]).reset_index(drop=True)

    def candidate_group_edges(
        self,
        target_groups: dict[int, tuple[int, ...]],
        *,
        metadata_for=None,
    ) -> pd.DataFrame:
        if self.backend == "gpu":
            from psvca.gpu.driver_gpu import run_gpu_batch

            return run_gpu_batch(
                driver=self,
                target_groups=target_groups,
                metadata_for=metadata_for,
            )
        return self._candidate_group_edges_cpu(target_groups, metadata_for=metadata_for)

    def _candidate_group_edges_cpu(
        self,
        target_groups: dict[int, tuple[int, ...]],
        *,
        metadata_for=None,
    ) -> pd.DataFrame:
        normalized_groups = {
            int(target): tuple(int(s) for s in group_sources if int(s) != int(target))
            for target, group_sources in target_groups.items()
        }
        normalized_groups = {
            target: group_sources
            for target, group_sources in normalized_groups.items()
            if group_sources
        }
        if self.n_jobs > 1 and len(normalized_groups) > 1:
            tasks = [
                {
                    "values": self.values,
                    "splits": self.splits,
                    "lookback": self.lookback,
                    "horizon": self.horizon,
                    "probe_config": self.probe_config,
                    "seed": self.seed,
                    "dataset": self.dataset,
                    "target": target,
                    "group_sources": group_sources,
                    "metadata_by_source": {
                        source: metadata_for(target, source) if metadata_for else None
                        for source in group_sources
                    },
                }
                for target, group_sources in normalized_groups.items()
            ]
            rows = _parallel_target_rows(_candidate_group_target_task, tasks, self.n_jobs)
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows).sort_values(["target", "source"]).reset_index(drop=True)

        rows = []
        for target, group_sources in normalized_groups.items():
            for source in group_sources:
                other_sources = tuple(s for s in group_sources if s != source)
                self.baseline_cache(int(target), other_sources)
            group_id = f"target_{int(target)}_top{len(group_sources)}"
            for source in group_sources:
                metadata = metadata_for(int(target), int(source)) if metadata_for else None
                rows.append(
                    self.probe_candidate_group_row(
                        target=int(target),
                        source=int(source),
                        group_sources=group_sources,
                        group_id=group_id,
                        metadata=metadata,
                    )
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["target", "source"]).reset_index(drop=True)

    def _dummy_target(self, source: int) -> int:
        if self.n_channels < 2:
            raise ValueError("at least two channels are required")
        return 0 if int(source) != 0 else 1

    def _own_design(self, target: int, split) -> DesignMatrix:
        return make_lagged_design(
            self.values,
            int(target),
            (),
            self.lookback,
            self.horizon,
            int(split.start),
            int(split.end),
            include_own=True,
        )

    def _source_design(
        self,
        target: int,
        source: int,
        split,
        *,
        values: np.ndarray | None = None,
    ) -> DesignMatrix:
        return make_lagged_design(
            self.values if values is None else values,
            int(target),
            (int(source),),
            self.lookback,
            self.horizon,
            int(split.start),
            int(split.end),
            include_own=False,
        )


def full_group_sources(n_channels: int, *, ref_group_cap: int | None = None) -> dict[int, tuple[int, ...]]:
    groups: dict[int, tuple[int, ...]] = {}
    for target in range(int(n_channels)):
        sources = tuple(source for source in range(int(n_channels)) if source != target)
        if ref_group_cap is not None:
            sources = sources[: int(ref_group_cap)]
        groups[target] = sources
    return groups


def workload_summary(edges: pd.DataFrame, *, mode: str, B: int) -> dict:
    if edges.empty:
        return {
            "n_edges": 0,
            "n_skipped": 0,
            "group_size": None,
            "group_size_min": None,
            "group_size_max": None,
            "group_size_mean": None,
            "svd_count_total": 0,
            "svd_count_own": 0,
            "svd_count_reduced": 0,
            "svd_count_full": 0,
            "svd_count_null": 0,
        }
    n_edges = int(len(edges))
    n_skipped = int(edges["skipped_null"].fillna(False).astype(bool).sum()) if "skipped_null" in edges else 0
    n_targets = int(edges["target"].nunique()) if "target" in edges else 0
    n_null = int((n_edges - n_skipped) * int(B))
    if "group_size" in edges:
        group_sizes = edges["group_size"].dropna().astype(int)
        unique_group_sizes = sorted(group_sizes.unique().tolist())
        group_size = int(unique_group_sizes[0]) if len(unique_group_sizes) == 1 else None
        group_size_min = int(group_sizes.min()) if not group_sizes.empty else None
        group_size_max = int(group_sizes.max()) if not group_sizes.empty else None
        group_size_mean = float(group_sizes.mean()) if not group_sizes.empty else None
    else:
        group_size = None
        group_size_min = None
        group_size_max = None
        group_size_mean = None

    if mode == "candidate_group":
        n_reduced = n_edges
        n_full = n_edges
    elif mode == "pairwise":
        n_reduced = 0
        n_full = n_edges
    else:
        raise ValueError(f"unsupported workload mode: {mode}")
    total = n_targets + n_reduced + n_full + n_null
    return {
        "n_edges": n_edges,
        "n_skipped": n_skipped,
        "group_size": group_size,
        "group_size_min": group_size_min,
        "group_size_max": group_size_max,
        "group_size_mean": group_size_mean,
        "svd_count_total": int(total),
        "svd_count_own": int(n_targets),
        "svd_count_reduced": int(n_reduced),
        "svd_count_full": int(n_full),
        "svd_count_null": int(n_null),
    }


def _stack_design(own: np.ndarray, blocks: dict[int, np.ndarray], sources: tuple[int, ...]) -> np.ndarray:
    if not sources:
        return own
    return np.column_stack([own, *(blocks[int(source)] for source in sources)])


def _result_row(result) -> dict:
    data = asdict(result)
    data.pop("delta_null", None)
    data.pop("alpha_null", None)
    data.setdefault("alpha_rule", "val_grid")
    return data


def _parallel_target_rows(fn, tasks: list[dict], n_jobs: int) -> list[dict]:
    max_workers = min(int(n_jobs), len(tasks))
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_pin_worker_blas,
    ) as executor:
        nested = list(executor.map(fn, tasks))
    return [row for rows in nested for row in rows]


def _pin_worker_blas() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"


def _driver_from_task(task: dict) -> CertificationDriver:
    return CertificationDriver(
        values=task["values"],
        splits=task["splits"],
        lookback=task["lookback"],
        horizon=task["horizon"],
        probe_config=task["probe_config"],
        seed=task["seed"],
        dataset=task["dataset"],
        n_jobs=1,
        backend="cpu",
    )


def _pairwise_target_task(task: dict) -> list[dict]:
    driver = _driver_from_task(task)
    target = int(task["target"])
    metadata_by_source = task["metadata_by_source"]
    rows = []
    for source in task["sources"]:
        rows.append(
            driver.probe_pairwise_row(
                target=target,
                source=int(source),
                metadata=metadata_by_source.get(int(source)),
            )
        )
    return rows


def _candidate_group_target_task(task: dict) -> list[dict]:
    driver = _driver_from_task(task)
    target = int(task["target"])
    group_sources = tuple(int(source) for source in task["group_sources"])
    metadata_by_source = task["metadata_by_source"]
    group_id = f"target_{target}_top{len(group_sources)}"
    for source in group_sources:
        other_sources = tuple(s for s in group_sources if s != source)
        driver.baseline_cache(target, other_sources)
    rows = []
    for source in group_sources:
        rows.append(
            driver.probe_candidate_group_row(
                target=target,
                source=int(source),
                group_sources=group_sources,
                group_id=group_id,
                metadata=metadata_by_source.get(int(source)),
            )
        )
    return rows
