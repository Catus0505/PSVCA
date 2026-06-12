from __future__ import annotations

import subprocess
from pathlib import Path

from psvca.config import PSVCAConfig, config_hash


def get_git_hash(repo_root: str | Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def make_run_id(cfg: PSVCAConfig) -> str:
    return f"{cfg.dataset}_pl{cfg.pred_len}_{cfg.tier}_{config_hash(cfg)}"


def ensure_run_dir(root: str | Path, run_id: str) -> Path:
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
