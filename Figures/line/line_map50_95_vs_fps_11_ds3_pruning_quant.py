#!/usr/bin/env python
"""
Line plot of mAP50-95 vs FPS for 11x DS3 baseline vs pruning+quantized engines.

Run:
  python Figures/line_map50_95_vs_fps_11_ds3_pruning_quant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from figure_save_dialog import prompt_save_figure
from ds3_line_utils import FPS_COLUMN, MAP_COLUMN, load_ds3_11_engine

CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"


def main() -> int:
    try:
        df = load_ds3_11_engine(CSV_PATH)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("Error: no DS3 11-series engine rows found.", file=sys.stderr)
        return 1

    df = df[df["kind"].isin(["baseline", "pruned_quant"])].copy()
    if df.empty:
        print("Error: no baseline/pruning+quantized DS3 engine rows found.", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap("tab10")

    lines = []
    baseline = df[df["kind"] == "baseline"].copy()
    if not baseline.empty:
        lines.append(("baseline", baseline))

    pq = df[df["kind"] == "pruned_quant"].copy()
    combos = (
        pq[["prune_pct", "quant_level"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["prune_pct", "quant_level"])
    )
    for _, row in combos.iterrows():
        pct = int(row["prune_pct"])
        q = str(row["quant_level"])
        group = pq[(pq["prune_pct"] == pct) & (pq["quant_level"] == q)].copy()
        if not group.empty:
            lines.append((f"p{pct}_{q}", group))

    for idx, (label, group) in enumerate(lines):
        group = group.sort_values("family_order")
        ax.plot(
            group[FPS_COLUMN],
            group[MAP_COLUMN],
            marker="o",
            linewidth=2,
            label=label,
            color=colors(idx),
        )
        for _, row in group.iterrows():
            ax.text(
                row[FPS_COLUMN],
                row[MAP_COLUMN],
                row["family"],
                fontsize=8,
                alpha=0.85,
                ha="left",
                va="bottom",
            )

    ax.set_xlabel("FPS (avg)")
    ax.set_ylabel(MAP_COLUMN)
    ax.set_title(f"{MAP_COLUMN} vs FPS (11x DS3 Baseline and Pruning+Quantized)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    plt.show()
    prompt_save_figure(fig, default_name="map50_95_vs_fps_11_ds3_pruning_quant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
