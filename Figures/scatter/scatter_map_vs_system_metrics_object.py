#!/usr/bin/env python
"""
Three scatterplots for DS3 object models:
  1) mAP50-95 vs GPU Util %
  2) mAP50-95 vs Power (W)
  3) mAP50-95 vs Temp (?C)

Layout:
  - IEEE double-column width
  - Three panels in one horizontal row

Run:
  python Figures/scatter_map_vs_system_metrics_object.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

MAP_COLUMN = "mAP50-95"
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
FILTER_TOKEN = "/object/"
DROP_PT_MODELS = True

METRIC_SPECS = [
    ("GPU Util %", "GPU Util %"),
    ("Power (W)", "Power (W)"),
    ("Temp (?C)", "Temp (C)"),
]

REQUIRED_COLS = ["Model", "Location", MAP_COLUMN] + [col for col, _ in METRIC_SPECS]

CATEGORY_ORDER = [
    ("baseline_pt", "Baseline", "#1f77b4"),
    ("baseline_engine", "Hardware Accel.", "#ff7f0e"),
    ("pruned_engine", "Pruned", "#2ca02c"),
    ("quant_engine_baseline", "Quantized", "#d62728"),
    ("quant_engine_pruned", "Quantized and Pruned", "#9467bd"),
    ("other", "Other", "#7f7f7f"),
]

IEEE_TWO_COL_WIDTH_IN = 7.16
FIG_WIDTH_IN = IEEE_TWO_COL_WIDTH_IN
FIG_HEIGHT_IN = 2.35
POINT_SIZE = 16


def _normalize_location(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def is_ds3(model: object, location: object) -> bool:
    m = str(model).lower()
    loc = _normalize_location(location)
    return "ds3" in m or "/ds3" in loc


def is_pt_model(model: object) -> bool:
    return str(model).lower().endswith(".pt")


def classify_row(model: object, location: object) -> str:
    loc = _normalize_location(location)
    m = str(model).lower()
    is_engine = ".engine" in m
    is_pt = m.endswith(".pt")

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


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["Location"].apply(lambda v: FILTER_TOKEN in _normalize_location(v))].copy()
    subset = subset[subset.apply(lambda r: is_ds3(r["Model"], r["Location"]), axis=1)].copy()
    if DROP_PT_MODELS:
        subset = subset[~subset["Model"].apply(is_pt_model)].copy()

    if subset.empty:
        raise ValueError("No DS3 object rows found.")

    numeric_cols = [MAP_COLUMN] + [col for col, _ in METRIC_SPECS]
    for col in numeric_cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    usable = subset.dropna(subset=numeric_cols).copy()
    if usable.empty:
        raise ValueError("No rows with numeric mAP and system metrics.")

    usable["category"] = usable.apply(lambda r: classify_row(r["Model"], r["Location"]), axis=1)
    return usable


def display_label(key: str, default_label: str) -> str:
    if DROP_PT_MODELS and key == "baseline_engine":
        return "Baseline"
    return default_label


def plot_grid(usable: pd.DataFrame) -> tuple[object, object]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        1,
        len(METRIC_SPECS),
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        dpi=300,
        sharey=True,
    )
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for ax, (metric_col, x_label) in zip(axes_flat, METRIC_SPECS):
        for key, label, color in CATEGORY_ORDER:
            group = usable[usable["category"] == key]
            if group.empty:
                continue
            ax.scatter(
                group[metric_col],
                group[MAP_COLUMN],
                alpha=0.82,
                s=POINT_SIZE,
                color=color,
                edgecolors="none",
            )

        ax.set_xlabel(x_label, fontsize=8)
        ax.set_title(f"{MAP_COLUMN} vs {x_label}", fontsize=8, pad=2)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes_flat[0].set_ylabel(MAP_COLUMN, fontsize=8)

    legend_handles = []
    for key, label, color in CATEGORY_ORDER:
        count = int((usable["category"] == key).sum())
        if count == 0:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markersize=5,
                label=f"{display_label(key, label)} (n={count})",
            )
        )

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            frameon=False,
            fontsize=6,
            bbox_to_anchor=(0.5, -0.03),
        )

    fig.suptitle(f"{MAP_COLUMN} vs System Metrics (Object DS3 Models)", fontsize=8, y=0.98)
    fig.subplots_adjust(wspace=0.28)
    fig.tight_layout(rect=[0, 0.09, 1, 0.93], pad=0.35)
    return fig, axes


def main() -> int:
    try:
        df = load_csv(CSV_PATH)
        usable = prepare_data(df)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Rows used:", len(usable))
    print("\nCounts by category:")
    print(usable["category"].value_counts())

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    fig, _ = plot_grid(usable)
    import matplotlib.pyplot as plt

    plt.show()
    prompt_save_figure(fig, default_name="object_map_vs_system_metrics_grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
