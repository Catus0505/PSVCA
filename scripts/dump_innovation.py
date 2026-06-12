from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psvca.base.innovation import compute_innovations_for_targets
from psvca.config import load_config
from psvca.data.loader import load_series
from psvca.io.artifacts import ensure_run_dir, make_run_id


def _describe_resid(name: str, resid) -> str:
    return f"{name}: n={len(resid)} mean={resid.mean():.6g} std={resid.std():.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--targets", nargs="+", type=int, required=True)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for dump_innovation.py") from exc

    cfg = load_config(args.config)
    loaded = load_series(cfg)
    out_dir = ensure_run_dir(Path("runs") / "innovation", make_run_id(cfg))
    results = compute_innovations_for_targets(
        values=loaded.values,
        channels=loaded.channels,
        splits=loaded.splits,
        targets=tuple(args.targets),
        lookback=cfg.lookback,
        horizon=cfg.pred_len,
        alphas=cfg.alpha_grid,
        seed=cfg.seed,
    )

    for result in results:
        print(f"target={result.target} channel={result.channel} alpha={result.alpha_own:.6g}")
        print("  " + _describe_resid("train_fit", result.resid_train_fit))
        print("  " + _describe_resid("val_alpha", result.resid_val_alpha))
        print("  " + _describe_resid("cert", result.resid_cert))

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(result.train_fit_indices, result.resid_train_fit, label="train_fit", linewidth=0.8)
        ax.plot(result.val_alpha_indices, result.resid_val_alpha, label="val_alpha", linewidth=0.8)
        ax.plot(result.cert_indices, result.resid_cert, label="cert", linewidth=0.8)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_title(f"{cfg.dataset} target {result.target} innovation")
        ax.set_xlabel("future index")
        ax.set_ylabel("residual")
        ax.legend(loc="best")
        fig.tight_layout()
        path = out_dir / f"target_{result.target}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  plot={path}")


if __name__ == "__main__":
    main()
