from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Range:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError(f"range bounds must be non-negative: {self}")
        if self.end < self.start:
            raise ValueError(f"range end must be >= start: {self}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Range") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class SplitRanges:
    original_train: Range
    original_val: Range
    original_test: Range
    pre_test: Range
    train_fit: Range
    val_alpha: Range
    cert: Range
    stability_blocks: tuple[Range, ...]


def _validate_range(name: str, r: Range, n_rows: int) -> None:
    if r.length <= 0:
        raise ValueError(f"{name} split is empty: {r}")
    if r.end > n_rows:
        raise ValueError(f"{name} split exceeds n_rows={n_rows}: {r}")


def compute_itransformer_borders(
    dataset_type: str,
    n_rows: int,
    seq_len: int,
    pred_len: int,
) -> tuple[Range, Range, Range]:
    del pred_len
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    if dataset_type == "ETT_hour":
        border1s = [
            0,
            12 * 30 * 24 - seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
    elif dataset_type == "ETT_minute":
        border1s = [
            0,
            12 * 30 * 24 * 4 - seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
    elif dataset_type in {"Custom", "synthetic"}:
        num_train = int(n_rows * 0.7)
        num_test = int(n_rows * 0.2)
        num_val = n_rows - num_train - num_test
        border1s = [0, num_train - seq_len, n_rows - num_test - seq_len]
        border2s = [num_train, num_train + num_val, n_rows]
    else:
        raise ValueError(f"unsupported dataset_type: {dataset_type}")

    ranges = tuple(Range(int(s), int(e)) for s, e in zip(border1s, border2s))
    for name, r in zip(("train", "val", "test"), ranges):
        _validate_range(name, r, n_rows)
    if ranges[1].start < 0 or ranges[2].start < 0:
        raise ValueError("seq_len is too large for the computed borders")
    return ranges


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3:
        raise ValueError("ratios must contain exactly three entries")
    if any(r <= 0 for r in ratios):
        raise ValueError(f"ratios must be positive: {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"ratios must sum to 1.0: {ratios}")


def _make_blocks(start: int, end: int, k_blocks: int) -> tuple[Range, ...]:
    if k_blocks <= 0:
        raise ValueError("k_blocks must be positive")
    total = end - start
    if total < k_blocks:
        raise ValueError("not enough pre-test rows for stability blocks")
    edges = [start + (total * i) // k_blocks for i in range(k_blocks + 1)]
    blocks = tuple(Range(edges[i], edges[i + 1]) for i in range(k_blocks))
    if any(block.length <= 0 for block in blocks):
        raise ValueError("empty stability block")
    return blocks


def carve_pretest_splits(
    original_train: Range,
    original_val: Range,
    original_test: Range,
    ratios: tuple[float, float, float],
    k_blocks: int,
) -> SplitRanges:
    _validate_ratios(ratios)
    if original_train.start != 0:
        raise ValueError("expected original_train to start at 0")
    if original_test.start < original_train.end:
        raise ValueError("test window starts before train ends")

    # iTransformer val/test windows include seq_len rows of context. Phase 0 treats
    # the test window start, including its context, as the first inaccessible row.
    pre_test = Range(original_train.start, original_test.start)
    if pre_test.length <= 0:
        raise ValueError("pre_test split is empty")
    if pre_test.end > original_val.end:
        raise ValueError("pre_test cannot extend beyond original validation end")

    n = pre_test.length
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_cert = n - n_train - n_val
    if min(n_train, n_val, n_cert) <= 0:
        raise ValueError("train_fit, val_alpha, and cert must be non-empty")

    train_fit = Range(pre_test.start, pre_test.start + n_train)
    val_alpha = Range(train_fit.end, train_fit.end + n_val)
    cert = Range(val_alpha.end, pre_test.end)
    if cert.overlaps(original_test):
        raise ValueError("cert split overlaps the iTransformer test window")

    blocks = _make_blocks(pre_test.start, pre_test.end, k_blocks)
    for name, r in {
        "train_fit": train_fit,
        "val_alpha": val_alpha,
        "cert": cert,
        **{f"stability_block_{i}": b for i, b in enumerate(blocks)},
    }.items():
        if r.start < pre_test.start or r.end > pre_test.end:
            raise ValueError(f"{name} is outside pre_test: {r}")
        if r.overlaps(original_test):
            raise ValueError(f"{name} overlaps original_test: {r}")

    return SplitRanges(
        original_train=original_train,
        original_val=original_val,
        original_test=original_test,
        pre_test=pre_test,
        train_fit=train_fit,
        val_alpha=val_alpha,
        cert=cert,
        stability_blocks=blocks,
    )
