from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.config import load_config
from psvca.data.loader import load_series
from psvca.data.registry import get_dataset_info


def _fmt_range(r) -> str:
    return f"[{r.start}, {r.end}) len={r.length}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    loaded = load_series(cfg)
    info = get_dataset_info(cfg.dataset)
    splits = loaded.splits

    print(f"dataset: {cfg.dataset}")
    print(f"dataset_type: {info.dataset_type}")
    print(f"n_rows_loaded: {loaded.values.shape[0]}")
    print(f"source_path: {loaded.source_path}")
    print(f"channel_count: {len(loaded.channels)}")
    print(f"original_train: {_fmt_range(splits.original_train)}")
    print(f"original_val: {_fmt_range(splits.original_val)}")
    print(f"original_test: {_fmt_range(splits.original_test)}")
    print(f"pre_test: {_fmt_range(splits.pre_test)}")
    print(f"train_fit: {_fmt_range(splits.train_fit)}")
    print(f"val_alpha: {_fmt_range(splits.val_alpha)}")
    print(f"cert: {_fmt_range(splits.cert)}")
    print("stability_blocks:")
    for idx, block in enumerate(splits.stability_blocks):
        print(f"  {idx}: {_fmt_range(block)}")


if __name__ == "__main__":
    main()
