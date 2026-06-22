from __future__ import annotations

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
    ) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.splits = splits
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.probe_config = probe_config
        self.seed = int(seed)
        self.dataset = str(dataset)
        self.n_jobs = int(n_jobs)
        self.n_channels = int(self.values.shape[1])
        self._own_designs: dict[int, tuple[DesignMatrix, DesignMatrix, DesignMatrix]] = {}
        self._own_caches: dict[int, BaselineFitCache] = {}
        self._source_designs: dict[int, DesignBlocks] = {}
        self._source_surrogates: dict[int, list[np.ndarray]] = {}

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
            own_train, own_val, own_cert = self.own_designs(target)
            self._own_caches[target] = fit_baseline_cache(
                sources=(),
                X_train=own_train.X,
                y_train=own_train.y,
                X_val=own_val.X,
                y_val=own_val.y,
                X_cert=own_cert.X,
                y_cert=own_cert.y,
                alphas=self.probe_config.alphas,
                alpha_rule=self.probe_config.alpha_rule,
                variance_eps=self.probe_config.variance_eps,
            )
        return self._own_caches[target]

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

    def surrogate_bank(self, *, target: int, source: int):
        source = int(source)
        target = int(target)
        for surrogate in self.phase_surrogates(source):
            s_values = self.values.copy()
            s_values[:, source] = surrogate
            yield (
                self._source_design(target, source, self.splits.train_fit, values=s_values).X,
                self._source_design(target, source, self.splits.val_alpha, values=s_values).X,
                self._source_design(target, source, self.splits.cert, values=s_values).X,
            )

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
            surrogate_bank=self.surrogate_bank(target=target, source=source),
            group_id=group_id,
            n_jobs=self.n_jobs,
            config=self.probe_config,
        )
        row = _result_row(result)
        if metadata:
            row.update(metadata)
        return row

    def pairwise_edges(self, targets, *, metadata_for=None) -> pd.DataFrame:
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
        rows = []
        for target, group_sources in target_groups.items():
            group_sources = tuple(int(s) for s in group_sources if int(s) != int(target))
            if not group_sources:
                continue
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


def _result_row(result) -> dict:
    data = asdict(result)
    data.pop("delta_null", None)
    data.pop("alpha_null", None)
    data.setdefault("alpha_rule", "val_grid")
    return data
