#!/usr/bin/env python
"""
Shared plotting logic for large single-metric object DS3 scatter charts.

This keeps the same filtering, category colors, and legend style as
scatter_map_vs_latency_object.py while swapping the x-axis metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

REQUIRED_BASE_COLS = ["Model", "Location"]
MAP_COLUMN = "mAP50-95"
MIN_MAP = 0.10
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
FILTER_TOKEN = "/object/"
DROP_PT_MODELS = False
FONT_FAMILY = "Times New Roman"
FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 4.2

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


def _normalize_location(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def is_ds3(model: object, location: object) -> bool:
    m = str(model).lower()
    loc = _normalize_location(location)
    return "ds3" in m or "/ds3" in loc


def classify_row(model: object, location: object) -> str:
    loc = _normalize_location(location)
    m = str(model).lower()
    is_engine = ".engine" in m
    is_pt = m.endswith(".pt")

    # Keep transformed .pt checkpoints grouped with transformed engines.
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


def is_pt_model(model: object) -> bool:
    return str(model).lower().endswith(".pt")


def load_csv(csv_path: Path, metric_col: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = REQUIRED_BASE_COLS + [metric_col, MAP_COLUMN]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def prepare_data(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    subset = df[df["Location"].apply(lambda value: FILTER_TOKEN in _normalize_location(value))].copy()
    subset = subset[subset.apply(lambda row: is_ds3(row["Model"], row["Location"]), axis=1)]
    if DROP_PT_MODELS:
        subset = subset[~subset["Model"].apply(is_pt_model)].copy()

    if subset.empty:
        raise ValueError("No DS3 object models found (location contains /object/ and DS3).")

    subset[MAP_COLUMN] = pd.to_numeric(subset[MAP_COLUMN], errors="coerce")
    subset[metric_col] = pd.to_numeric(subset[metric_col], errors="coerce")
    numeric = subset.dropna(subset=[MAP_COLUMN, metric_col]).copy()
    usable = numeric[numeric[MAP_COLUMN] >= MIN_MAP].copy()

    dropped_non_numeric = len(subset) - len(numeric)
    if dropped_non_numeric:
        print(f"Warning: dropped {dropped_non_numeric} rows with non-numeric {MAP_COLUMN} or {metric_col}.")

    dropped_low_map = len(numeric) - len(usable)
    if dropped_low_map:
        print(f"Warning: dropped {dropped_low_map} rows with {MAP_COLUMN} < {MIN_MAP:.2f}.")

    if usable.empty:
        raise ValueError(f"No object rows with numeric mAP/{metric_col} and {MAP_COLUMN} >= {MIN_MAP:.2f}.")

    usable["category"] = usable.apply(lambda row: classify_row(row["Model"], row["Location"]), axis=1)
    return usable


def plot_metric(usable: pd.DataFrame, *, metric_col: str, x_label: str) -> tuple[object, object]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)

    present_categories = []
    for key, _, color in CATEGORY_ORDER:
        group = usable[usable["category"] == key]
        if group.empty:
            continue
        present_categories.append((key, color))
        ax.scatter(
            group[metric_col],
            group[MAP_COLUMN],
            alpha=0.8,
            s=16,
            color=color,
        )

    ax.set_xlabel(x_label, fontsize=8)
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
    return fig, ax


def run_metric_plot(*, metric_col: str, x_label: str, default_name: str) -> int:
    try:
        df = load_csv(CSV_PATH, metric_col)
        usable = prepare_data(df, metric_col)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Rows used:", len(usable))
    print("Metric:", metric_col)
    print("\nCounts by category:")
    print(usable["category"].value_counts())

    try:
        fig, _ = plot_metric(usable, metric_col=metric_col, x_label=x_label)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    plt.show()
    prompt_save_figure(fig, default_name=default_name)
    return 0
