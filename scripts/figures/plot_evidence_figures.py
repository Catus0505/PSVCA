from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"required figure input is missing: {path}")
    return path


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.alpha": 0.22,
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.sort(values[np.isfinite(values)])
    y = np.arange(1, len(vals) + 1, dtype=np.float64) / max(len(vals), 1)
    return vals, y


def plot_fig1(input_root: Path, output_dir: Path, tier: str) -> None:
    recovery = pd.read_parquet(_require(input_root / f"fig1_synthetic_recovery_{tier}.parquet"))
    pvals = pd.read_parquet(_require(input_root / f"fig1_pvalue_calibration_{tier}.parquet"))
    p_summary = pd.read_csv(_require(input_root / "fig1_pvalue_calibration_summary.tsv"), sep="\t")
    r_summary = pd.read_csv(_require(input_root / "fig1_synthetic_recovery_summary.tsv"), sep="\t")

    rates = (
        recovery.groupby("edge_type")["admitted_candidate"]
        .mean()
        .reindex(["TRUE", "NULL", "DECOY"])
    )
    counts = recovery.groupby("edge_type").size().reindex(["TRUE", "NULL", "DECOY"])
    B = int(pvals["B"].iloc[0])
    ks_p = float(p_summary["ks_pvalue"].iloc[0])
    fpr = float(p_summary["empirical_fpr_005"].iloc[0])
    decoy_failed = bool(r_summary["decoy_oracle_failed"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.20, wspace=0.26)
    ax = axes[0]
    colors = ["#2F6B4F", "#6F7785", "#B36B2C"]
    ax.bar(rates.index, rates.to_numpy(dtype=float), color=colors, width=0.64)
    ax.axhline(0.10, color="#444444", linestyle="--", linewidth=1.2, label="q = 0.10")
    ax.set_ylim(0.0, max(0.18, float(np.nanmax(rates.to_numpy(dtype=float))) * 1.25))
    ax.set_ylabel("admission rate")
    ax.set_title("A. Synthetic planted-edge recovery")
    for idx, (label, rate) in enumerate(rates.items()):
        ax.text(idx, rate + 0.01, f"n={int(counts[label])}", ha="center", va="bottom", fontsize=9)
    ax.text(
        0.0,
        -0.18,
        f"tier={tier}, B={B}; pairwise candidate, no FDR/stability yet\n"
        f"no test split used for certification; decoy oracle failed={decoy_failed}",
        transform=ax.transAxes,
        fontsize=8.5,
    )

    ax = axes[1]
    x, y = _ecdf(pvals["p_value"].to_numpy(dtype=float))
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.1, linestyle="--", label="Uniform")
    ax.step(x, y, where="post", color="#275DAD", linewidth=2.0, label="empirical ECDF")
    ax.axvline(0.05, color="#B22222", linestyle=":", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("p-value under null")
    ax.set_ylabel("ECDF")
    ax.set_title("B. Null p-value calibration")
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.04,
        0.82,
        f"KS p={ks_p:.3g}\nFPR@0.05={fpr:.3g}\nno test split used for certification",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#D6D6D6", "alpha": 0.9, "pad": 4},
    )
    fig.suptitle("Fig.1 Instrument validation", fontsize=15)
    _save(fig, output_dir, "fig1_instrument_validation")
    plt.close(fig)


def _matrix(df: pd.DataFrame, field: str) -> np.ndarray:
    n = int(max(df["target"].max(), df["source"].max()) + 1)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    for row in df.itertuples():
        mat[int(row.target), int(row.source)] = float(getattr(row, field))
    return mat


def _short_summary_lines(text: str, *, limit: int = 3) -> list[str]:
    if not isinstance(text, str) or not text or text == "nan":
        return []
    lines = []
    for item in text.split("; ")[:limit]:
        item = item.replace("coherence_peak=", "coh=")
        item = item.replace("aligned_gain=", "gain=")
        item = item.replace("diff=", "diff=")
        lines.append(item)
    return lines


def _format_summary_box(s: pd.Series) -> str:
    high_assoc = _short_summary_lines(str(s["top_high_association_rejected_edges"]))
    asym = _short_summary_lines(str(s["top_direction_asymmetry_pairs"]))
    lines = ["C. Summary: association != value", "", "High-association but rejected:"]
    lines.extend(f"  - {line}" for line in high_assoc)
    lines.append("")
    lines.append("Direction asymmetry:")
    lines.extend(f"  - {line}" for line in asym)
    return "\n".join(lines)


def plot_fig2(input_root: Path, output_dir: Path, tier: str, dataset: str, pred_len: int) -> None:
    detail = input_root / f"fig2_real_value_association_{dataset}_pl{pred_len}_{tier}.parquet"
    df = pd.read_parquet(_require(detail))
    summary = pd.read_csv(_require(input_root / "fig2_real_value_association_summary.tsv"), sep="\t")
    row = summary[
        (summary["dataset"] == dataset)
        & (summary["pred_len"] == pred_len)
        & (summary["tier"] == tier)
    ]
    if row.empty:
        raise SystemExit(f"summary row not found for dataset={dataset}, pred_len={pred_len}, tier={tier}")
    s = row.iloc[0]
    B = int(df["B"].iloc[0])
    assoc = _matrix(df, "coherence_peak")
    gain = _matrix(df, "aligned_gain")
    candidate = _matrix(df, "certified_candidate")

    fig = plt.figure(figsize=(16.4, 8.8))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[3.7, 1.35],
        width_ratios=[1.08, 1.08, 1.70],
        hspace=0.34,
        wspace=0.32,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.07)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    im0 = ax0.imshow(assoc, cmap="viridis", vmin=0.0, vmax=np.nanmax(assoc))
    ax0.set_title("A1. Association matrix (peak coherence)", fontsize=11, pad=10)
    ax0.set_xlabel("source")
    ax0.set_ylabel("target")
    ax0.set_xticks(np.arange(assoc.shape[1]))
    ax0.set_yticks(np.arange(assoc.shape[0]))
    fig.colorbar(im0, ax=ax0, fraction=0.050, pad=0.025)

    vmax = np.nanmax(np.abs(gain))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    im1 = ax1.imshow(gain, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax1.set_title("A2. Directed value matrix (aligned gain)", fontsize=11, pad=10)
    ax1.set_xlabel("source")
    ax1.set_ylabel("target")
    ax1.set_xticks(np.arange(gain.shape[1]))
    ax1.set_yticks(np.arange(gain.shape[0]))
    yy, xx = np.where(candidate == 1.0)
    ax1.scatter(xx, yy, s=78, facecolors="none", edgecolors="black", linewidths=1.4)
    fig.colorbar(im1, ax=ax1, fraction=0.050, pad=0.025)
    ax1.text(
        0.02,
        0.02,
        "association is near-symmetric;\nvalue is directed",
        transform=ax1.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#D6D6D6", "alpha": 0.88, "pad": 3},
    )

    rejected = ~df["certified_candidate"].to_numpy(dtype=bool)
    accepted = df["certified_candidate"].to_numpy(dtype=bool)
    ax2.scatter(
        df.loc[rejected, "coherence_peak"],
        df.loc[rejected, "aligned_gain"],
        c="#777777",
        s=32,
        label="rejected",
        alpha=0.75,
    )
    ax2.scatter(
        df.loc[accepted, "coherence_peak"],
        df.loc[accepted, "aligned_gain"],
        c="#B22222",
        s=46,
        marker="D",
        label="certified_candidate",
        alpha=0.9,
    )
    ax2.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax2.set_xlabel("association foil: peak coherence")
    ax2.set_ylabel("aligned_gain")
    ax2.set_title("B. Association vs directed value")
    ax2.grid(True, alpha=0.18)
    ax2.legend(frameon=False)
    ax2.text(
        0.04,
        0.96,
        f"Spearman $\\rho$={float(s['spearman_coherence_peak_aligned_gain']):.3g}\n"
        f"tier={tier}, B={B}\nno test split used for certification\n"
        "pairwise candidate, no FDR/stability yet",
        transform=ax2.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#D6D6D6", "alpha": 0.92, "pad": 4},
    )

    ax3.axis("off")
    ax3.text(
        0.01,
        0.96,
        _format_summary_box(s),
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        linespacing=1.22,
        bbox={"facecolor": "#FAFAFA", "edgecolor": "#CFCFCF", "alpha": 1.0, "pad": 8},
    )
    fig.suptitle(f"Fig.2 Value vs association ({dataset}, pred_len={pred_len})", fontsize=15)
    _save(fig, output_dir, "fig2_value_vs_association")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="runs/figures")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--fig1-tier", required=True)
    parser.add_argument("--fig2-tier", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    args = parser.parse_args()
    _style()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    plot_fig1(input_root, output_dir, args.fig1_tier)
    plot_fig2(input_root, output_dir, args.fig2_tier, args.dataset, args.pred_len)
    print(f"wrote {output_dir / 'fig1_instrument_validation.svg'}")
    print(f"wrote {output_dir / 'fig1_instrument_validation.png'}")
    print(f"wrote {output_dir / 'fig2_value_vs_association.svg'}")
    print(f"wrote {output_dir / 'fig2_value_vs_association.png'}")


if __name__ == "__main__":
    main()
