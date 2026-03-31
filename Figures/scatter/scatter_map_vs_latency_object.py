#!/usr/bin/env python
"""
Scatter plot of mAP vs latency for object models only.

Run:
  python Figures/scatter_map_vs_latency_object.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

REQUIRED_COLS = ["Model", "Location", "Latency ms"]
MAP_COLUMN = "mAP50-95"
MIN_MAP = 0.10
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
FILTER_TOKEN = "/object/"
IEEE_ONE_COL_WIDTH_IN = 3.5
IEEE_ONE_COL_HEIGHT_IN = 2.5
DROP_PT_MODELS = False
FONT_FAMILY = "Times New Roman"

CATEGORY_ORDER = [
    ("baseline_pt", "Baseline", "#1f77b4"),
    ("baseline_engine", "Hardware Accel.", "#ff7f0e"),
    ("pruned_engine", "Pruned", "#2ca02c"),
    ("quant_engine_baseline", "Quantized", "#d62728"),
    ("quant_engine_pruned", "Quantized, Pruned", "#9467bd"),
    ("distilled", "Distillation", "#8c564b"),
    ("distilled_pruned", "Distilled, Pruned", "#17becf"),
    ("distilled_pruned_quantized", "Distilled, Pruned, Quantized", "#bcbd22"),
    ("distilled_quantized", "Distilled, Quantized", "#e377c2"),
    ("other", "Other", "#7f7f7f"),
]


def _normalize_location(value: str) -> str:
    return str(value).replace("\\", "/").lower()


def is_ds3(model: str, location: str) -> bool:
    m = str(model).lower()
    loc = _normalize_location(location)
    return "ds3" in m or "/ds3" in loc


def classify_row(model: str, location: str) -> str:
    loc = _normalize_location(location)
    m = str(model).lower()
    is_engine = ".engine" in m
    is_pt = m.endswith(".pt")

    # Treat transformed .pt checkpoints the same as transformed engine artifacts.
    # This prevents pruned/quantized .pt rows from being bucketed as "other".
    if "/distillation_pruning_quantization/" in loc and (is_engine or is_pt):
        return "distilled_pruned_quantized"
    if "/distillation_pruning/" in loc and (is_engine or is_pt):
        return "distilled_pruned"
    if "/distilled-quantized/" in loc and (is_engine or is_pt):
        return "distilled_quantized"
    if "/distillation/" in loc and (is_engine or is_pt):
        return "distilled"
    if "/pruning_quantization/" in loc and (is_engine or is_pt):
        return "quant_engine_pruned"
    if "/quantization/" in loc and (is_engine or is_pt):
        return "quant_engine_baseline"
    if "/pruning/" in loc and (is_engine or is_pt):
        return "pruned_engine"
    if "/baseline/" in loc:
        if is_pt:
            return "baseline_pt"
        if is_engine:
            return "baseline_engine"
    return "other"


def is_pt_model(model: str) -> bool:
    return str(model).lower().endswith(".pt")


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS + [MAP_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def main() -> int:
    try:
        df = load_csv(CSV_PATH)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    subset = df[df["Location"].apply(lambda v: FILTER_TOKEN in _normalize_location(v))].copy()
    subset = subset[subset.apply(lambda r: is_ds3(r["Model"], r["Location"]), axis=1)]
    if DROP_PT_MODELS:
        subset = subset[~subset["Model"].apply(is_pt_model)].copy()
    if subset.empty:
        print("Error: no DS3 object models found (location contains /object/ and DS3).", file=sys.stderr)
        return 1

    subset[MAP_COLUMN] = pd.to_numeric(subset[MAP_COLUMN], errors="coerce")
    subset["Latency ms"] = pd.to_numeric(subset["Latency ms"], errors="coerce")
    numeric = subset.dropna(subset=[MAP_COLUMN, "Latency ms"]).copy()
    usable = numeric[numeric[MAP_COLUMN] >= MIN_MAP].copy()

    dropped_non_numeric = len(subset) - len(numeric)
    if dropped_non_numeric:
        print(f"Warning: dropped {dropped_non_numeric} rows with non-numeric {MAP_COLUMN} or Latency ms.")

    dropped_low_map = len(numeric) - len(usable)
    if dropped_low_map:
        print(f"Warning: dropped {dropped_low_map} rows with {MAP_COLUMN} < {MIN_MAP:.2f}.")

    if usable.empty:
        print(
            f"Error: no object rows with numeric mAP/Latency and {MAP_COLUMN} >= {MIN_MAP:.2f}.",
            file=sys.stderr,
        )
        return 1

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FormatStrFormatter
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    usable["category"] = usable.apply(lambda r: classify_row(r["Model"], r["Location"]), axis=1)

    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDTH_IN, IEEE_ONE_COL_HEIGHT_IN), dpi=300)
    present_categories = []
    for key, _, color in CATEGORY_ORDER:
        group = usable[usable["category"] == key]
        if group.empty:
            continue
        present_categories.append((key, color))
        ax.scatter(
            group["Latency ms"],
            group[MAP_COLUMN],
            alpha=0.8,
            s=16,
            color=color,
        )

    ax.set_xlabel("Latency ms", fontsize=8)
    ax.set_ylabel(MAP_COLUMN, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    label_by_key = {key: label for key, label, _ in CATEGORY_ORDER}
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=5, label=label_by_key[key])
        for key, color in present_categories
    ]
    if handles:
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=6,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=3,
            borderaxespad=0.0,
            columnspacing=0.8,
            handletextpad=0.3,
        )

    fig.tight_layout(pad=0.08)
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.33, top=0.995)
    plt.show()
    prompt_save_figure(fig, default_name="object_map_vs_latency")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
