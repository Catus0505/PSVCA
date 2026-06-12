from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.config import load_config
from psvca.data.loader import load_series
from psvca.nulls.phase_surrogate import make_surrogate_bank_for_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True, type=int)
    parser.add_argument("--B", required=True, type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    loaded = load_series(cfg)
    if args.source < 0 or args.source >= loaded.values.shape[1]:
        raise SystemExit(f"source index out of bounds: {args.source}")
    results = make_surrogate_bank_for_source(
        loaded.values[:, args.source],
        source_idx=args.source,
        B=args.B,
        seed=cfg.seed,
        dataset=cfg.dataset,
        split="pre_test",
        cache_dir="surrogate_cache",
    )
    hits = sum(result.cache_hit for result in results)
    misses = len(results) - hits
    print(f"cache_hits={hits} cache_misses={misses}")
    for result in results:
        status = "hit" if result.cache_hit else "miss"
        print(f"{status}: {result.cache_path}")


if __name__ == "__main__":
    main()
