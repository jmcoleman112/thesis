#!/usr/bin/env python
"""
Compare mAP50-95 vs FPS for two groups of models using
research/FAKED_MODEL_SUMMARIES_NOT_ACCURATE.csv (default).

Edit GROUP_A / GROUP_B lists below as needed, then run:
  python Figures/compare_map50_95_vs_fps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

REQUIRED_COLS = ["Model", "mAP50-95", "FPS (avg)"]

# Edit these as needed
GROUP_A_LABEL = "YOLOv11"
GROUP_B_LABEL = "YOLOv26"
GROUP_A = [
    "11x_DS3_baseline.engine",
    "11l_DS3_baseline.engine",
    "11m_DS3_baseline.engine",
    "11s_DS3_baseline.engine",
    "11n_DS3_baseline.engine"
]
GROUP_B = [
    "26x_DS3_baseline.engine",
    "26l_DS3_baseline.engine",
    "26m_DS3_baseline.engine",
    "26s_DS3_baseline.engine",
    "26n_DS3_baseline.engine"
    ]

CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def subset_group(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    if not models:
        raise ValueError("Model list is empty.")

    subset = df[df["Model"].isin(models)].copy()
    if subset.empty:
        raise ValueError("No matching models found for list.")

    # Coerce numeric columns
    subset["mAP50-95"] = pd.to_numeric(subset["mAP50-95"], errors="coerce")
    subset["FPS (avg)"] = pd.to_numeric(subset["FPS (avg)"], errors="coerce")

    subset = subset.dropna(subset=["mAP50-95", "FPS (avg)"])
    if subset.empty:
        raise ValueError("No rows with numeric mAP50-95 and FPS (avg).")

    # Sort by FPS to create a line
    subset = subset.sort_values("FPS (avg)")
    return subset


def main() -> int:
    try:
        df = load_csv(CSV_PATH)
        group_a = subset_group(df, GROUP_A)
        group_b = subset_group(df, GROUP_B)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Warn about missing models
    missing_a = [m for m in GROUP_A if m not in set(group_a["Model"])]
    missing_b = [m for m in GROUP_B if m not in set(group_b["Model"])]
    if missing_a:
        print(f"Warning: group A missing models: {missing_a}")
    if missing_b:
        print(f"Warning: group B missing models: {missing_b}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        group_a["FPS (avg)"],
        group_a["mAP50-95"],
        marker="o",
        linewidth=2,
        alpha=0.7,
        label=GROUP_A_LABEL,
    )
    ax.plot(
        group_b["FPS (avg)"],
        group_b["mAP50-95"],
        marker="o",
        linewidth=2,
        alpha=0.7,
        label=GROUP_B_LABEL,
    )

    ax.set_xlabel("FPS (avg)")
    ax.set_ylabel("mAP50-95")
    ax.set_title("mAP50-95 vs FPS")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Small labels next to each dot: first three letters of model name
    for _, row in group_a.iterrows():
        label = str(row["Model"])[:3]
        ax.text(
            row["FPS (avg)"],
            row["mAP50-95"],
            label,
            fontsize=8,
            alpha=0.85,
            ha="left",
            va="bottom",
        )

    for _, row in group_b.iterrows():
        label = str(row["Model"])[:3]
        ax.text(
            row["FPS (avg)"],
            row["mAP50-95"],
            label,
            fontsize=8,
            alpha=0.85,
            ha="left",
            va="bottom",
        )

    fig.tight_layout()

    plt.show()
    prompt_save_figure(fig, default_name="map50_95_vs_fps")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
